"""Fine-tune old classifier on HA with exact runtime resize and safe augmentations."""
import argparse
import json
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms as T
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

from classifier_metrics import classification_metrics, safety_fitness, dual_safety_fitness, LABELS

cv2.setNumThreads(1)


def runtime_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255.0


def safe_augment(image):
    # No crop/translation/rotation/erasing/mixup: those could change TARGET-slot labels.
    if np.random.random() < .5:
        image = cv2.flip(image, 1)
    result = image.astype(np.float32)
    contrast, exposure = np.random.uniform(.9,1.1), np.random.uniform(.85,1.15)
    mean = result.mean(axis=(0,1), keepdims=True)
    result = ((result-mean)*contrast+mean)*exposure
    result = np.clip(result,0,255)/255
    result = np.power(result, np.random.uniform(.9,1.1))*255
    if np.random.random() < .15:
        ramp = np.linspace(np.random.uniform(.92,1.08), np.random.uniform(.92,1.08), result.shape[1])
        result *= ramp[None,:,None]
    if np.random.random() < .2:
        result += np.random.normal(0, np.random.uniform(.3,1.5), result.shape)
    result = np.clip(result,0,255).astype(np.uint8)
    if np.random.random() < .1:
        result = cv2.GaussianBlur(result, (3,3), np.random.uniform(.3,.6))
    return result


class RuntimeDataset(ClassificationDataset):
    def __init__(self, *args, augment=False, **kwargs):
        super().__init__(*args, augment=augment, **kwargs)
        self.runtime_augment = augment
        # Portable checkpoint metadata for generic PIL inference (approximate only).
        # Actual train/val/ONNX tests always use cv2 runtime_tensor, never this PIL transform.
        self.torch_transforms = T.Compose([T.Resize((224,224)), T.ToTensor()])
        manifest_path=Path(self.root).parent/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        self.legacy_validation_paths={str((manifest_path.parent/r['image']).resolve())
                                      for r in manifest['records'] if r.get('origin')=='legacy_val'}

    def __getitem__(self, index):
        filename, label = self.samples[index][:2]
        image = cv2.imread(filename)
        if image is None:
            raise ValueError(filename)
        if self.runtime_augment:
            image = safe_augment(image)
        return {'img':runtime_tensor(image), 'cls':label,
                'domain':int(str(Path(filename).resolve()) in self.legacy_validation_paths)}


class MacroF1Validator(ClassificationValidator):
    def build_dataset(self, img_path):
        return RuntimeDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_stats(self):
        result = super().get_stats()
        targets = torch.cat(self.targets).cpu().numpy()
        predictions = torch.cat(self.pred)[:,0].cpu().numpy()
        metrics = classification_metrics(targets, predictions)
        result.update({'metrics/macro_f1':metrics['macro_f1'],
                       'metrics/ng_recall':metrics['per_class']['ng']['recall'],
                       'metrics/no_disk_recall':metrics['per_class']['no_disk']['recall'],
                       'metrics/unsafe_false_ok_rate':metrics['unsafe_false_ok_rate'],
                       'fitness':metrics['macro_f1']})
        return result


class SafetyValidator(MacroF1Validator):
    def get_stats(self):
        result = super().get_stats()
        metrics = classification_metrics(torch.cat(self.targets).cpu().numpy(),
                                         torch.cat(self.pred)[:,0].cpu().numpy())
        result['fitness'] = safety_fitness(metrics)
        result['metrics/ng_as_ok'] = metrics['ng_as_ok']
        result['metrics/ng_as_no_disk'] = metrics['ng_as_no_disk']
        result['metrics/no_disk_as_ok'] = metrics['no_disk_as_ok']
        result['metrics/ok_recall'] = metrics['per_class']['ok']['recall']
        return result


class DualSafetyValidator(MacroF1Validator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.domains=[]
        manifest=json.loads((Path(self.args.data)/'manifest.json').read_text(encoding='utf-8'))
        self.safety_gates=manifest.get('validation_safety_gates')

    def update_metrics(self, preds, batch):
        super().update_metrics(preds,batch)
        self.domains.append(batch['domain'].cpu())

    def get_stats(self):
        result=super().get_stats()
        targets=torch.cat(self.targets).numpy()
        predictions=torch.cat(self.pred)[:,0].numpy()
        domains=torch.cat(self.domains).numpy()
        metrics={name:classification_metrics(targets[domains==i],predictions[domains==i])
                 for i,name in enumerate(('ha','legacy'))}
        assert metrics['ha']['count']==400 and metrics['legacy']['count']==2770
        for name,m in metrics.items():
            for key in ('accuracy','macro_f1','ng_as_ok','ng_as_no_disk','no_disk_as_ok'):
                result[f'metrics/{name}_{key}']=m[key]
        result['fitness']=dual_safety_fitness(metrics['ha'],metrics['legacy'],self.safety_gates)
        return result


class RuntimeTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return RuntimeDataset(root=img_path, args=self.args, augment=mode=='train', prefix=mode)

    def get_validator(self):
        self.loss_names = ['loss']
        return MacroF1Validator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode != 'train':
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank != -1:
            raise ValueError('This audited HA trainer supports single GPU only')
        dataset = self.build_dataset(dataset_path, mode)
        counts = np.bincount([s[1] for s in dataset.samples], minlength=3)
        assert (counts>0).all() and dataset.base.classes == LABELS
        weights = [1/np.sqrt(counts[s[1]]) for s in dataset.samples]
        manifest = json.loads((Path(self.args.data)/'manifest.json').read_text(encoding='utf-8'))
        if manifest['domain'] in ('HA_with_fault_priority_replay','HA_dual_validation_fault_replay'):
            origins = {str((Path(self.args.data)/r['image']).resolve()):r['origin']
                       for r in manifest['records'] if r['split']=='train'}
            groups = [origins[str(Path(s[0]).resolve())] for s in dataset.samples]
            group_counts = {group:np.bincount([s[1] for s,g in zip(dataset.samples,groups) if g==group],minlength=3)
                            for group in ('ha','legacy_train')}
            raw_weights = [([1.35,1.15,1.0][s[1]]/np.sqrt(group_counts['ha'][s[1]]) if g=='ha'
                            else manifest.get('legacy_class_sampling',[.75,.15,.10])[s[1]]/group_counts['legacy_train'][s[1]])
                           for s,g in zip(dataset.samples,groups)]
            totals = {group:sum(w for w,g in zip(raw_weights,groups) if g==group)
                      for group in ('ha','legacy_train')}
            ha_probability=manifest.get('ha_sampling_probability',.85)
            weights = [w/totals[g]*(ha_probability if g=='ha' else 1-ha_probability) for w,g in zip(raw_weights,groups)]
        generator = torch.Generator().manual_seed(self.args.seed)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
        return InfiniteDataLoader(dataset, batch_size=min(batch_size,len(dataset)), sampler=sampler,
                                  num_workers=self.args.workers, pin_memory=True,
                                  worker_init_fn=seed_worker, generator=generator)


class SafetyTrainer(RuntimeTrainer):
    def get_validator(self):
        self.loss_names = ['loss']
        return SafetyValidator(self.test_loader,self.save_dir,args=copy(self.args),_callbacks=self.callbacks)


class DualSafetyTrainer(RuntimeTrainer):
    def get_validator(self):
        self.loss_names=['loss']
        return DualSafetyValidator(self.test_loader,self.save_dir,args=copy(self.args),_callbacks=self.callbacks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--name', default='ha_only')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr0',type=float,default=.0003)
    parser.add_argument('--dropout',type=float,default=.1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['domain'] in ('human_reviewed_HA_only','HA_with_fault_priority_replay','HA_dual_validation_fault_replay')
    assert manifest['training_authorized_by_user']
    if (args.project/args.name).exists():
        raise FileExistsError('Training results are never overwritten')
    model = YOLO(str(args.pretrained.resolve()))
    assert [model.names[i] for i in range(3)]==LABELS
    trainer = {'human_reviewed_HA_only':RuntimeTrainer,'HA_with_fault_priority_replay':SafetyTrainer,
               'HA_dual_validation_fault_replay':DualSafetyTrainer}[manifest['domain']]
    model.train(trainer=trainer, data=str(args.data.resolve()), epochs=args.epochs,
                imgsz=224, batch=args.batch, workers=2, device=0, seed=1005, deterministic=True,
                optimizer='AdamW', lr0=args.lr0, lrf=.1, cos_lr=True, weight_decay=.0005,
                warmup_epochs=5, warmup_bias_lr=args.lr0, patience=20, dropout=args.dropout,
                amp=False, freeze=None, auto_augment=None, erasing=0, scale=0,
                fliplr=.5, flipud=0, bgr=0, close_mosaic=0,
                project=str(args.project.resolve()), name=args.name, exist_ok=False,
                plots=True, verbose=False)
    best = model.trainer.best
    # Canonical artifact consumed by runtime; unchanged preprocessing and class order.
    exported = YOLO(str(best)).export(format='onnx', dynamic=True, simplify=True, opset=13,
                                     imgsz=224, batch=1, device='cpu')
    print(json.dumps({'best_pt':str(best), 'onnx':str(exported),
                      'selection':manifest.get('selection_policy','HA validation macro-F1; test not used for selection')}, indent=2))


if __name__ == '__main__':
    main()
