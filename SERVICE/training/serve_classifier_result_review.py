"""Review disagreement labels without changing frozen evaluation/training artifacts."""
import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

LABELS = {'ng', 'no_disk', 'ok'}
TEMPLATE = Path(__file__).with_name('classifier_result_review.html')


def review_payload(root):
    records_path, report_path = root/'per_image.json', root/'report.json'
    records_bytes, report_bytes = records_path.read_bytes(), report_path.read_bytes()
    digest = hashlib.sha256(records_bytes + b'\0' + report_bytes).hexdigest()
    items = []
    for record in json.loads(records_bytes):
        predictions = record['predictions']
        if not any(prediction['label'] != record['label'] for prediction in predictions.values()):
            continue
        domain = 'legacy_val' if record['id'].startswith('legacy_') else 'ha_' + record['split']
        native = f'errors/{domain}/{record["id"]}.png'
        native_path = (root/native).resolve()
        if not native_path.is_relative_to(root.resolve()) or not native_path.is_file():
            raise ValueError('Missing or unsafe native crop: ' + record['id'])
        items.append({'id': record['id'], 'original_label': record['label'], 'domain': domain,
                      'image': native, 'path': record['path'], 'predictions': predictions,
                      'frame': record.get('frame'), 'region': record.get('region'),
                      'slot': record.get('slot')})
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('Duplicate evaluation IDs')
    return digest, items, json.loads(report_bytes)


def validate_changes(values, items):
    if not isinstance(values, list) or not 1 <= len(values) <= 2000:
        raise ValueError('Expected 1..2000 decisions')
    checked = []
    for value in values:
        if not isinstance(value, dict) or value.get('id') not in items:
            raise ValueError('Unknown disagreement crop')
        action = value.get('action', 'save')
        if action not in {'save', 'reset'}:
            raise ValueError('Invalid action')
        if action == 'save' and value.get('label') not in LABELS:
            raise ValueError('Invalid label')
        notes = value.get('notes', '')
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError('Notes must be text, at most 2000 characters')
        checked.append({'id': value['id'], 'action': action,
                        'label': value.get('label') if action == 'save' else None, 'notes': notes})
    if len({value['id'] for value in checked}) != len(checked):
        raise ValueError('Duplicate crop in request')
    return checked


class ResultReviewHandler(SimpleHTTPRequestHandler):
    lock = Lock()
    output_dir = None
    digest = None
    items = None
    report = None

    def state(self):
        path = self.output_dir/'label_review'/'decisions.json'
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            if value.get('evaluation_sha256') != self.digest:
                raise ValueError('Corrections belong to another evaluation')
            return value
        return {'schema_version': 1, 'evaluation_sha256': self.digest, 'revision': 0,
                'decisions': {}, 'history': [], 'training_authorized': False,
                'original_evaluation_unchanged': True,
                'review_scope': 'original disagreements from either model; not a blind full-dataset audit'}

    def respond(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in {'/', '/review'}:
            body = TEMPLATE.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        elif path == '/api/review':
            with self.lock:
                self.respond({'evaluation_sha256': self.digest, 'items': list(self.items.values()),
                              'state': self.state(), 'report': self.report})
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/api/save':
            self.respond({'error': 'Unknown endpoint'}, 404)
            return
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',
                                     f'http://localhost:{self.server.server_port}'}:
            self.respond({'error': 'Origin rejected'}, 403)
            return
        if self.headers.get_content_type() != 'application/json':
            self.respond({'error': 'JSON required'}, 415)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2_000_000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or body.get('evaluation_sha256') != self.digest:
                raise ValueError('Stale or different evaluation')
            checked = validate_changes(body.get('decisions'), self.items)
            with self.lock:
                state = self.state()
                if type(body.get('revision')) is not int or body['revision'] != state['revision']:
                    self.respond({'error': 'Trang khác đã sửa nhãn. Reload trước khi sửa tiếp.',
                                  'state': state}, 409)
                    return
                now = datetime.now(timezone.utc).isoformat()
                state['revision'] += 1
                for change in checked:
                    identifier = change['id']
                    item = self.items[identifier]
                    previous = state['decisions'].get(identifier)
                    state['history'].append(dict(change, original_label=item['original_label'],
                        previous_decision=previous, domain=item['domain'], updated_at=now,
                        revision=state['revision'], source='human_disagreement_review_ui'))
                    if change['action'] == 'reset':
                        state['decisions'].pop(identifier, None)
                    else:
                        state['decisions'][identifier] = {
                            'label': change['label'], 'original_label': item['original_label'],
                            'notes': change['notes'], 'domain': item['domain'], 'reviewed': True,
                            'updated_at': now, 'revision': state['revision'],
                            'source': 'human_disagreement_review_ui'}
                state['updated_at'] = now
                folder = self.output_dir/'label_review'
                folder.mkdir(exist_ok=True)
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                                      suffix='.json', delete=False) as handle:
                        temp_path = Path(handle.name)
                        json.dump(state, handle, ensure_ascii=False, indent=2)
                    temp_path.replace(folder/'decisions.json')
                finally:
                    if temp_path is not None and temp_path.exists():
                        temp_path.unlink()
            self.respond({'saved': len(checked), 'state': state})
        except (ValueError, KeyError, TypeError) as error:
            self.respond({'error': str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8777)
    args = parser.parse_args()
    root = args.evaluation.resolve()
    digest, items, report = review_payload(root)
    ResultReviewHandler.output_dir = root
    ResultReviewHandler.digest = digest
    ResultReviewHandler.items = {item['id']: item for item in items}
    ResultReviewHandler.report = report
    ResultReviewHandler.state(ResultReviewHandler.__new__(ResultReviewHandler))
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                partial(ResultReviewHandler, directory=str(root)))
    print(f'Correction review: http://127.0.0.1:{args.port}/ | {len(items)} disagreements', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
