"""Review a completed classifier run's error cases against live canonical labels."""
import argparse
import csv
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from classifier_dataset import CanonicalDataset, LABELS, REGISTRY, StaleRevision, read

PAGE = Path(__file__).with_name('classifier_error_review.html')


class ErrorReview:
    def __init__(self, run, store):
        self.store = store
        snapshot = read(run / 'input_snapshot/manifest.json')
        self.revision = snapshot['canonical_revision']
        validation = {r['id']: r for r in snapshot['records'] if r['split'] == 'val'}
        with (run / 'validation_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            predictions = list(csv.DictReader(stream))
        if len(predictions) != len(validation) or {r['id'] for r in predictions} != set(validation):
            raise ValueError('Predictions do not match snapshot validation IDs')
        self.cases = {}
        for row in predictions:
            frozen = validation[row['id']]
            if (row['truth'] != frozen['label'] or row['path'] != frozen['image']
                    or row['best111'] not in LABELS or row['learned'] not in LABELS):
                raise ValueError('Prediction/snapshot mismatch: ' + row['id'])
            if row['truth'] == row['best111'] == row['learned']:
                continue
            self.cases[row['id']] = {
                **row, 'sha256': frozen['sha256'],
                'frozen_updated_at': frozen.get('canonical_updated_at'),
            }

    def live_case(self, case, record):
        matches = record is not None and record['sha256'] == case['sha256']
        return {
            'label': record['label'] if matches else None,
            'editable': matches and record['split'] == 'val' and record['label'] in LABELS,
            'reviewed_after_run': matches and record.get('human_reviewed') is True
            and record.get('canonical_updated_at') != case['frozen_updated_at'],
        }

    def state(self):
        manifest = self.store.manifest()
        records = {r['id']: r for r in manifest['records'] + manifest.get('excluded_records', [])}
        return {
            'revision': manifest['canonical_revision'],
            'frozen_revision': self.revision,
            'cases': [{**case, 'current': self.live_case(case, records.get(case['id']))}
                      for case in self.cases.values()],
        }

    def save(self, body):
        if not isinstance(body, dict):
            raise ValueError('Dữ liệu gửi lên không hợp lệ.')
        changes = body.get('changes')
        if changes is None:
            changes = [{'id': body['id'], 'label': body['label']}]
        if not isinstance(changes, list) or not 1 <= len(changes) <= len(self.cases):
            raise ValueError('Danh sách nhãn cần lưu không hợp lệ.')
        edits, seen = [], set()
        for change in changes:
            identifier, label = change['id'], change['label']
            if identifier not in self.cases or identifier in seen or label not in LABELS:
                raise ValueError('Chọn case trong danh sách và nhãn NG, Trống hoặc OK.')
            seen.add(identifier)
            edits.append({'id': identifier, 'label': label, 'expected_split': 'val',
                          'expected_sha256': self.cases[identifier]['sha256'],
                          'notes': f'Rà case lỗi từ validation revision {self.revision}'})
        result = self.store.save(edits, body['revision'])
        updated = {r['id']: self.live_case(self.cases[r['id']], r)
                   for r in result['updated'] if r['id'] in self.cases}
        return {'revision': result['revision'], 'updated': updated, 'saved': result['saved']}


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, review, **kwargs):
        self.review = review
        super().__init__(*args, **kwargs)

    def send_payload(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def json(self, value, status=200):
        self.send_payload(json.dumps(value, ensure_ascii=False).encode('utf-8'),
                          'application/json; charset=utf-8', status)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/errors.html':
            self.send_payload(PAGE.read_bytes(), 'text/html; charset=utf-8')
        elif path == '/api/error-review':
            self.json(self.review.state())
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/api/error-review/label':
            self.json({'error': 'Không tìm thấy trang lưu nhãn.'}, 404)
            return
        hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host')
        if host not in hosts or self.headers.get('Origin') != f'http://{host}':
            self.json({'error': 'Mở trang sửa nhãn từ địa chỉ localhost.'}, 403)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if self.headers.get_content_type() != 'application/json' or not 0 < size <= 16384:
                raise ValueError('Dữ liệu gửi lên không hợp lệ.')
            self.json(self.review.save(json.loads(self.rfile.read(size))))
        except StaleRevision:
            self.json({'error': 'Dataset vừa đổi. Đã giữ lựa chọn để mày kiểm tra rồi bấm lưu lại.'}, 409)
        except (ValueError, TypeError, KeyError) as exc:
            self.json({'error': str(exc)}, 400)
        except (OSError, RuntimeError) as exc:
            self.log_error('Canonical save failed: %s', exc)
            self.json({'error': 'Chưa lưu được. Tải lại nhãn hiện tại rồi thử lại.'}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    run = args.run.resolve()
    if not read(run / 'completion.json')['complete']:
        parser.error('The classifier run must be complete')
    review = ErrorReview(run, CanonicalDataset(read(REGISTRY)['path']))
    handler = partial(ReviewHandler, review=review, directory=str(run))
    with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
        print(f'Canonical error review: http://127.0.0.1:{args.port}/errors.html', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
