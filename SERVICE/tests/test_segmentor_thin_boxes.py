import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
try:
    import torch
    from train_segmentor_thin_boxes import ThinBoxTrainer, widen_thin_boxes
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None,'Optional training dependencies torch/ultralytics unavailable')
class ThinBoxTests(unittest.TestCase):
    def test_long_thin_envelope_widens_without_mutation(self):
        boxes = torch.tensor([[.5,.5,.01,.9],[.5,.5,.2,.9],[.5,.5,.01,.1]])
        original = boxes.clone()
        result = widen_thin_boxes(boxes,512,512)
        self.assertTrue(torch.equal(boxes,original))
        self.assertAlmostEqual(float(result[0,2]),24/512)
        self.assertTrue(torch.allclose(result[1:],boxes[1:]))
        self.assertTrue(torch.allclose(result[:,:2],boxes[:,:2]))
        self.assertTrue(torch.equal(result[:,3],boxes[:,3]))

    def test_boundary_and_empty(self):
        result = widen_thin_boxes(torch.tensor([[.005,.5,.005,.9]]),512,512)
        self.assertGreaterEqual(float(result[0,0]-result[0,2]/2),0)
        self.assertLessEqual(float(result[0,0]+result[0,2]/2),1)
        self.assertEqual(widen_thin_boxes(torch.empty((0,4)),512,512).shape,(0,4))

    def test_batch_preprocess_does_not_change_mask_pixels(self):
        trainer = object.__new__(ThinBoxTrainer)
        trainer.device = torch.device('cpu')
        trainer.args = SimpleNamespace(multi_scale=False)
        masks = torch.zeros((1,256,256),dtype=torch.uint8)
        masks[:,5:250,125:127] = 1
        batch = {'img':torch.zeros((1,3,512,512),dtype=torch.uint8),
                 'bboxes':torch.tensor([[.5,.5,.01,.9]]),'masks':masks.clone()}
        result = trainer.preprocess_batch(batch)
        self.assertTrue(torch.equal(result['masks'],masks))
        self.assertAlmostEqual(float(result['bboxes'][0,2]),24/512)


if __name__=='__main__':
    unittest.main()
