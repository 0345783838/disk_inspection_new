"""Edit the one canonical classifier dataset directly; never train or deploy."""
import argparse
import csv
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

try:
    from .classifier_dataset import CanonicalDataset, REGISTRY, StaleRevision, read
    from .classifier_diagnostic_review import live_diagnostics
except ImportError:
    from classifier_dataset import CanonicalDataset, REGISTRY, StaleRevision, read
    from classifier_diagnostic_review import live_diagnostics


class DatasetHandler(SimpleHTTPRequestHandler):
    store = None
    pairs = []
    template = Path(__file__).with_name('classifier_canonical_review.html')

    def json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ('/', '/index.html'):
            data = self.template.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == '/api/state':
            self.json(dict(self.store.state(), pairs=self.pairs))
        elif path == '/api/evaluation':
            try:
                self.json(live_diagnostics(self.store.manifest(), self.store.root))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.json({'available': False, 'error': 'Không tải được đánh giá rà lỗi: ' + str(exc)})
        elif path == '/api/pairs.csv':
            state = self.store.state()
            records = {r['id']: r for r in state['records']}
            stream = io.StringIO(newline='')
            writer = csv.writer(stream)
            writer.writerow(['pair', 'id_a', 'label_a', 'image_a', 'id_b', 'label_b', 'image_b'])
            for pair in self.pairs:
                if pair['a'] not in records or pair['b'] not in records:
                    continue
                a, b = records[pair['a']], records[pair['b']]
                writer.writerow([pair['id'], a['id'], a['label'], a['image'], b['id'], b['label'], b['image']])
            data = ('\ufeff' + stream.getvalue()).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/csv; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename="classifier_label_pairs.csv"')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != '/api/labels':
            self.json({'error': 'Use the canonical label editor'}, 404)
            return
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}:
            self.json({'error': 'Origin rejected'}, 403)
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 2_000_000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            result = self.store.save(body.get('changes'), body.get('revision'))
            self.json(result)
        except StaleRevision as exc:
            self.json({'error': str(exc)}, 409)
        except (ValueError, TypeError, KeyError) as exc:
            self.json({'error': str(exc)}, 400)
        except Exception as exc:
            self.json({'error': 'Dataset update failed; transaction recovery attempted: ' + str(exc)}, 500)


class RetiredReviewHandler(SimpleHTTPRequestHandler):
    target = None

    def do_GET(self):
        # All old portal URLs lead to the one editor, retaining only query filters.
        query = urlsplit(self.path).query
        target = self.target + ('?' + query if query else '')
        self.send_response(302)
        self.send_header('Location', target)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

    def do_POST(self):
        data = json.dumps({'error': 'Old editor retired. Open ' + self.target + ' to edit the canonical dataset.'}).encode()
        self.send_response(409)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8791)
    parser.add_argument('--legacy-ports', default='')
    args = parser.parse_args()
    registry = read(REGISTRY)
    store = CanonicalDataset(registry['path'])
    source = read(registry['pair_candidates'])
    pairs = [{'id': f'pair_{i+1:03}', 'a': r['a'], 'b': r['b'], 'priority': r.get('priority', False),
              'distance': r['rmse96']} for i, r in enumerate(source)]
    active_ids = {r['id'] for r in store.state()['records']}
    pairs = [p for p in pairs if p['a'] in active_ids and p['b'] in active_ids]
    class Handler(DatasetHandler):
        pass
    Handler.store, Handler.pairs = store, pairs
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(store.root)))
    for port in [int(v) for v in args.legacy_ports.split(',') if v]:
        class Redirect(RetiredReviewHandler):
            target = f'http://127.0.0.1:{args.port}/'
        legacy = ThreadingHTTPServer(('127.0.0.1', port), Redirect)
        threading.Thread(target=legacy.serve_forever, daemon=True).start()
    print(f'Canonical dataset: {store.root} | http://127.0.0.1:{args.port}/ | {len(pairs)} pairs', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
