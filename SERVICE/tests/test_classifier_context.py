import tempfile,unittest
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics.nn.tasks import ClassificationModel
from training.classifier_context import ContextClassifier,make_contexts,save_model,load_model,state_sha
from training.train_classifier_context import metrics,score
from training.build_classifier_context import holdout,find_project,PROJECT

class ContextTests(unittest.TestCase):
    def test_project_root_from_nested_frozen_snapshot(self):
        self.assertEqual(find_project(PROJECT/'.work/example_run/code_snapshot/build_classifier_context.py'),PROJECT)
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,control=False):
        torch.manual_seed(1031)
        return ContextClassifier(ClassificationModel(cfg='yolo11n-cls.yaml',nc=3,verbose=False),control).eval()
    def test_query_mask_and_top_rotation_preserve_raw_pixels(self):
        im=np.zeros((50,120,3),np.uint8);im[:,:,0]=np.arange(120,dtype=np.uint8)[None,:]
        rects=[[10,10,30,30],[30,10,50,30],[50,10,70,30]]
        bottom,description=make_contexts(im,rects,'bottom');top,_=make_contexts(im,rects,'top')
        self.assertEqual(bottom[0].shape,(224,224,4));self.assertEqual(set(np.unique(bottom[0][:,:,3])),{0,255})
        native=np.zeros((20,100),np.uint8);native[:,40:60]=255
        expected=cv2.resize(cv2.rotate(native,cv2.ROTATE_180),(224,224),interpolation=cv2.INTER_NEAREST)
        self.assertTrue(np.array_equal(top[1][:,:,3],expected))
        self.assertEqual(description[0]['pitch'],20.)
    def test_zero_initial_correction_and_missing_context_exact_parent(self):
        model=self.model();images=torch.rand(2,3,224,224);context=torch.rand(2,4,224,224)
        with torch.no_grad():
            features,logits=model.local(images)
            self.assertTrue(torch.equal(model(images,context),logits.softmax(1)))
            model.fusion[-1].weight.normal_()
            self.assertTrue(torch.equal(model(images,torch.zeros_like(context)),logits.softmax(1)))
    def test_control_ignores_rgb_and_roundtrip_export(self):
        import onnxruntime as ort
        model=self.model(True)
        with torch.no_grad():model.fusion[-1].weight.normal_(std=.01)
        images=torch.rand(2,3,224,224);a=torch.rand(2,4,224,224);b=a.clone();b[:,:3]=torch.rand_like(b[:,:3])
        with torch.no_grad():self.assertTrue(torch.equal(model(images,a),model(images,b)))
        with tempfile.TemporaryDirectory() as tmp:
            pt=Path(tmp)/'model.pt';save_model(pt,model,epoch=3);restored=load_model(pt)
            with torch.no_grad():self.assertTrue(torch.equal(model(images,a),restored(images,a)))
            onnx=Path(tmp)/'model.onnx'
            torch.onnx.export(restored,(images,a),str(onnx),input_names=['local','context'],output_names=['probabilities'],dynamic_axes={'local':{0:'batch'},'context':{0:'batch'},'probabilities':{0:'batch'}},opset_version=17)
            options=ort.SessionOptions();options.intra_op_num_threads=2;session=ort.InferenceSession(str(onnx),sess_options=options,providers=['CPUExecutionProvider'])
            with torch.no_grad():expected=restored(images,a).numpy()
            actual=session.run(None,dict(local=images.numpy(),context=a.numpy()))[0]
            self.assertLess(float(np.max(abs(actual-expected))),1e-3);self.assertTrue(np.array_equal(actual.argmax(1),expected.argmax(1)))
            for count in [1,3]:
                local=torch.rand(count,3,224,224);ctx=torch.rand(count,4,224,224)
                with torch.no_grad():expected=restored(local,ctx).numpy()
                actual=session.run(None,dict(local=local.numpy(),context=ctx.numpy()))[0]
                self.assertLess(float(np.max(abs(actual-expected))),1e-3)
    def test_training_never_changes_parent_buffers_or_weights(self):
        model=self.model();before=state_sha(model.frozen_state());model.train();images=torch.rand(2,3,224,224)
        with torch.no_grad():features,logits=model.local(images)
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.001)
        out,aux,_=model.cached(features,logits,torch.rand(2,4,224,224));loss=torch.nn.functional.cross_entropy(out,torch.tensor([0,1]))+torch.nn.functional.cross_entropy(aux,torch.tensor([0,1]));loss.backward();optimizer.step()
        self.assertEqual(state_sha(model.frozen_state()),before)
    def test_primary_metrics_without_legacy_and_score_ranking(self):
        rows=[dict(origin='old_ha',camera=2,label='ng'),dict(origin='new_factory',camera=1,label='ok')]
        anchor=metrics(rows,[dict(label='ok'),dict(label='ok')]);better=metrics(rows,[dict(label='ng'),dict(label='ok')])
        self.assertGreater(score(better,anchor),score(anchor,anchor))
    def test_capture_quarantine_honors_canonical_roles_and_frame_disjointness(self):
        from copy import deepcopy
        rows=[]
        for origin,bursts in [('old_ha',2),('new_factory',4)]:
            for group in range(bursts):
                cameras=[group+1] if origin=='old_ha' else [1,2]
                for camera in cameras:
                    for label in ['ng','no_disk','ok']:
                        for split in ['train','val']:
                            rows.append(dict(id=f'{origin}/{group}/{camera}/{label}/{split}',origin=origin,camera=camera,label=label,split=split,frame=f'{origin}/frame{group}/cam{camera}',collection_burst=f'{origin}_burst_{group}'))
        before=deepcopy(rows);result=holdout(rows);by_id={r['id']:r for r in rows}
        self.assertEqual(rows,before)
        self.assertTrue(all(by_id[i]['split']=='train' for i in result['fit_ids']))
        self.assertTrue(all(by_id[i]['split']=='val' for i in result['primary_validation_ids']))
        self.assertFalse(set(result['fit_frames'])&set(result['primary_frames']))
        self.assertTrue(set(result['quarantined_train_ids']).isdisjoint(result['fit_ids']))

if __name__=='__main__':unittest.main()
