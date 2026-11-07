"""Local human label review; proposals immutable, decisions separate, no training."""
import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

ALLOWED = {'ok', 'ng', 'no_disk', 'uncertain', 'exclude', 'unreviewed'}


def validate_decisions(values, item_ids):
    if not isinstance(values, list) or not 1 <= len(values) <= 2000:
        raise ValueError('Expected 1..2000 decisions')
    checked = []
    for value in values:
        if not isinstance(value, dict) or value.get('id') not in item_ids:
            raise ValueError('Unknown crop')
        if value.get('label') not in ALLOWED:
            raise ValueError('Invalid label')
        checked.append({'id': value['id'], 'label': value['label'],
                        'notes': str(value.get('notes', ''))[:2000]})
    if len({v['id'] for v in checked}) != len(checked):
        raise ValueError('Duplicate crop in request')
    return checked


class ReviewHandler(SimpleHTTPRequestHandler):
    lock = Lock()
    output_dir = None
    digest = None
    item_ids = None

    def state(self):
        path = self.output_dir/'decisions.json'
        if path.exists():
            state = json.loads(path.read_text(encoding='utf-8'))
            if state.get('items_sha256') != self.digest:
                raise ValueError('Decisions belong to another dataset')
            return state
        return {'schema_version': 1, 'items_sha256': self.digest, 'decisions': {},
                'training_authorized': False}

    def respond(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/state':
            with self.lock:
                self.respond(self.state())
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
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 2_000_000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            if body.get('items_sha256') != self.digest:
                raise ValueError('Stale or different dataset')
            checked = validate_decisions(body.get('decisions'), self.item_ids)
            with self.lock:
                state = self.state()
                now = datetime.now(timezone.utc).isoformat()
                for value in checked:
                    identifier = value.pop('id')
                    if value['label'] == 'unreviewed':
                        state['decisions'].pop(identifier, None)
                    else:
                        state['decisions'][identifier] = dict(value, updated_at=now, source='human_review_ui')
                state['updated_at'] = now
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.output_dir,
                                                  suffix='.json', delete=False) as handle:
                    json.dump(state, handle, ensure_ascii=False, indent=2)
                    temp = Path(handle.name)
                temp.replace(self.output_dir/'decisions.json')
            self.respond({'saved': len(checked), 'state': state})
        except (ValueError, KeyError, TypeError) as error:
            self.respond({'error': str(error)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8776)
    args = parser.parse_args()
    registry = Path(__file__).resolve().parents[2] / '.work/ACTIVE_CLASSIFIER_DATASET.json'
    if registry.exists():
        raise RuntimeError('Use the single canonical dataset editor at http://127.0.0.1:8791/; old decisions-only editors are retired.')
    output = args.output.resolve()
    payload_path = output/'items.json'
    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    ReviewHandler.output_dir = output
    ReviewHandler.digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    ReviewHandler.item_ids = {i['id'] for i in payload['items']}
    existing = output/'decisions.json'
    if existing.exists():
        assert json.loads(existing.read_text(encoding='utf-8'))['items_sha256'] == ReviewHandler.digest
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(ReviewHandler, directory=str(output)))
    print(f'Review: http://127.0.0.1:{server.server_port}/ | {len(payload["items"])} crops', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
