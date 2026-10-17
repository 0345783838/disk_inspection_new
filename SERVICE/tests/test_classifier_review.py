"""Review persistence tests run only on temporary fixtures, never real labels."""
import hashlib
import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from training.serve_classifier_review import ReviewHandler, validate_decisions


class ClassifierReviewTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(validate_decisions([{'id':'a','label':'ok'}], {'a'})[0]['label'], 'ok')
        for values in ([], None, [{'id':'unknown','label':'ok'}], [{'id':'a','label':'bad'}],
                       [{'id':'a','label':'ok'}, {'id':'a','label':'ng'}]):
            with self.assertRaises(ValueError):
                validate_decisions(values, {'a'})

    def test_http_save_atomic_and_no_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposals = b'{"fixture":true}'
            (root/'items.json').write_bytes(proposals)
            digest = hashlib.sha256(proposals).hexdigest()
            class Handler(ReviewHandler):
                output_dir = root
                item_ids = {'a', 'b'}
                def log_message(self, *args):
                    pass
            Handler.digest = digest
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(root)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def post(values, **kwargs):
                body = json.dumps({'items_sha256':kwargs.get('digest', digest), 'decisions':values}).encode()
                request = Request(base+'/api/save', data=body, headers={'Content-Type':'application/json',
                                  'Origin':kwargs.get('origin', base)})
                with urlopen(request) as response:
                    return json.load(response)
            try:
                with urlopen(base+'/api/state') as response:
                    self.assertEqual(json.load(response)['decisions'], {})
                post([{'id':'a','label':'ok','notes':'human fixture'}])
                post([{'id':'a','label':'ng'}, {'id':'b','label':'uncertain'}])
                saved = json.loads((root/'decisions.json').read_text())
                self.assertEqual(saved['decisions']['a']['label'], 'ng')
                self.assertEqual(saved['decisions']['b']['label'], 'uncertain')
                self.assertFalse(saved['training_authorized'])
                self.assertEqual((root/'items.json').read_bytes(), proposals)
                prior = (root/'decisions.json').read_bytes()
                for kwargs, values in (({'origin':'http://example.invalid'}, [{'id':'a','label':'ok'}]),
                                       ({'digest':'stale'}, [{'id':'a','label':'ok'}]),
                                       ({}, [{'id':'a','label':'ok'},{'id':'missing','label':'ok'}])):
                    with self.assertRaises(HTTPError):
                        post(values, **kwargs)
                    self.assertEqual((root/'decisions.json').read_bytes(), prior)
                post([{'id':'a','label':'unreviewed'}])
                self.assertNotIn('a', json.loads((root/'decisions.json').read_text())['decisions'])
                self.assertEqual(len(list(root.glob('*.json'))), 2)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()


if __name__ == '__main__':
    unittest.main()
