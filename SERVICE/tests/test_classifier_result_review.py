"""Fixture-only label corrections; never write user labels during tests."""
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

from training.serve_classifier_result_review import ResultReviewHandler, review_payload, validate_changes


class ClassifierResultReviewTests(unittest.TestCase):
    def fixture(self, root):
        records = []
        for identifier, label, baseline, candidate in [('a', 'ng', 'ng', 'ok'),
                                                      ('b', 'ok', 'ng', 'ok'),
                                                      ('c', 'ok', 'ok', 'ok')]:
            records.append({'id': identifier, 'label': label, 'split': 'test', 'path': 'fixture-only',
                            'predictions': {'baseline': {'label': baseline, 'confidence': .7},
                                            'candidate': {'label': candidate, 'confidence': .9}}})
            folder = root/'errors'/'ha_test'
            folder.mkdir(parents=True, exist_ok=True)
            (folder/(identifier+'.png')).write_bytes(b'fixture native bytes')
        (root/'per_image.json').write_text(json.dumps(records), encoding='utf-8')
        (root/'report.json').write_text('{"domains":{}}', encoding='utf-8')
        return review_payload(root)

    def test_payload_only_original_disagreements_and_bound_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest, items, _ = self.fixture(root)
            self.assertEqual([item['id'] for item in items], ['a', 'b'])
            self.assertEqual(items[0]['original_label'], 'ng')
            (root/'report.json').write_text('{"domains":{},"changed":true}')
            self.assertNotEqual(review_payload(root)[0], digest)
            (root/'errors'/'ha_test'/'a.png').unlink()
            with self.assertRaises(ValueError):
                review_payload(root)

    def test_validation(self):
        self.assertEqual(validate_changes([{'id': 'a', 'label': 'ng'}], {'a'})[0]['action'], 'save')
        self.assertIsNone(validate_changes([{'id': 'a', 'action': 'reset'}], {'a'})[0]['label'])
        for values in [[], None, [{'id': 'unknown', 'label': 'ng'}], [{'id': 'a', 'label': 'bad'}],
                       [{'id': 'a', 'label': 'ng', 'notes': 'x'*2001}],
                       [{'id': 'a', 'label': 'ng', 'action': 'train'}],
                       [{'id': 'a', 'label': 'ng'}, {'id': 'a', 'label': 'ok'}]]:
            with self.assertRaises(ValueError):
                validate_changes(values, {'a'})

    def test_http_history_reset_stale_tabs_and_immutable_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digest, items, report = self.fixture(root)
            frozen = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob('*') if path.is_file()}
            class Handler(ResultReviewHandler):
                output_dir = root
                def log_message(self, *args):
                    pass
            Handler.digest, Handler.items, Handler.report = digest, {i['id']: i for i in items}, report
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(root)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f'http://127.0.0.1:{server.server_port}'
            def post(values, revision, **kwargs):
                data = json.dumps({'evaluation_sha256': kwargs.get('digest', digest),
                                   'revision': revision, 'decisions': values}).encode()
                request = Request(base+'/api/save', data=data, headers={
                    'Content-Type': kwargs.get('content_type', 'application/json'),
                    'Origin': kwargs.get('origin', base)})
                with urlopen(request) as response:
                    return json.load(response)
            try:
                with urlopen(base+'/') as response:
                    self.assertIn('sửa'.encode(), response.read().lower())
                with urlopen(base+'/api/review') as response:
                    self.assertEqual(json.load(response)['state']['revision'], 0)
                # Merely loading the UI must not create any decisions file.
                self.assertFalse((root/'label_review').exists())
                first = post([{'id': 'a', 'label': 'ok', 'notes': 'human fixture'}], 0)['state']
                self.assertEqual(first['decisions']['a']['original_label'], 'ng')
                saved_path = root/'label_review'/'decisions.json'
                prior = saved_path.read_bytes()
                invalid = [([{'id': 'a', 'label': 'ng'}], 0, {}, 409),
                           ([{'id': 'a', 'label': 'ng'}], 1, {'origin': 'http://evil.invalid'}, 403),
                           ([{'id': 'a', 'label': 'ng'}], 1, {'digest': 'wrong'}, 400),
                           ([{'id': 'a', 'label': 'ng'}], 1, {'content_type': 'text/plain'}, 415),
                           ([{'id': 'a', 'label': 'ng'}, {'id': 'c', 'label': 'ng'}], 1, {}, 400)]
                for values, revision, kwargs, code in invalid:
                    with self.assertRaises(HTTPError) as error:
                        post(values, revision, **kwargs)
                    self.assertEqual(error.exception.code, code)
                    self.assertEqual(saved_path.read_bytes(), prior)
                post([{'id': 'a', 'label': 'no_disk'}, {'id': 'b', 'label': 'ok'}], 1)
                final = post([{'id': 'a', 'action': 'reset'}], 2)['state']
                self.assertNotIn('a', final['decisions'])
                self.assertEqual(final['decisions']['b']['original_label'], 'ok')
                self.assertEqual(len(final['history']), 4)
                self.assertEqual(final['history'][1]['previous_decision']['label'], 'ok')
                self.assertEqual(final['history'][-1]['previous_decision']['label'], 'no_disk')
                self.assertFalse(final['training_authorized'])
                self.assertTrue(final['original_evaluation_unchanged'])
                with urlopen(base+'/api/review') as response:
                    self.assertEqual(json.load(response)['state']['revision'], 3)
                for path, expected in frozen.items():
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
                self.assertEqual(list((root/'label_review').iterdir()), [saved_path])
                Handler.digest = 'other-evaluation'
                with self.assertRaises(ValueError):
                    Handler.state(Handler.__new__(Handler))
            finally:
                server.shutdown()
                server.server_close()
                worker.join()


if __name__ == '__main__':
    unittest.main()
