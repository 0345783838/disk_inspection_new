"""Nano classifier with a dense groove/entry/trace branch for a matched pilot.

The ordinary forward accepts runtime RGB/stretch224 tensors and returns three
class probabilities. ``return_aux=True`` returns ``(logits, auxiliary)``.
Annotations, coordinate masks and losses belong to the caller. No annotation,
groove estimate or visibility truth is required as an inference input.

Both control and supervised methods must instantiate the SAME model/seed. The
only experimental difference should be the caller's auxiliary-loss weights.
Unknown or estimated coordinates must never be given an observed-point loss.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel


LABELS = ['ng', 'no_disk', 'ok']
HEATMAP_NAMES = ['left_peak', 'right_peak', 'entry1', 'entry2', 'trace1', 'trace2']
VISIBILITY_NAMES = ['visible', 'partial', 'occluded', 'unknown']
OWNERSHIP_NAMES = ['target', 'left_neighbor', 'right_neighbor']
FORMAT = 'groove_ownership_nano_v1'


def tensor(image):
    """The existing SERVICE input transform, without a crop or normalization change."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.


def state_sha(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


class GrooveOwnershipClassifier(nn.Module):
    """Preserved nano head plus a zero-initialized, geometry-conditioned residual.

    With the current YOLO11n parent, backbone blocks0--7 are frozen and blocks8/9
    plus the original classifier train. All BatchNorm running statistics stay
    frozen. The early block4 map is128x28x28; trainable late features are resized
    from256x7x7 and fused with it. Thus auxiliary losses can influence trainable
    late features, and the geometry branch participates in the primary decision.

    Heatmap channel pairs are ordered instances, not two rims of one disk.
    The data loader must keep entry1/trace1 and entry2/trace2 associated, and
    transform left/right peak, neighbor ownership and instance ordering on flips.
    """
    def __init__(self, parent, train_last_blocks=2, tap_index=4,
                 decoder_channels=32, embedding_dim=64, head_dropout_p=None):
        super().__init__()
        if any(getattr(layer, 'f', -1) != -1 for layer in parent.model):
            raise ValueError('Expected sequential YOLO classifier')
        parent = deepcopy(parent).float().eval()
        self.parent_yaml = deepcopy(parent.yaml)
        self.names = dict(enumerate(LABELS))
        if dict(parent.names) != self.names:
            raise ValueError('Expected parent class order ng/no_disk/ok')
        self.backbone = nn.Sequential(*list(parent.model.children())[:-1])
        self.classify = list(parent.model.children())[-1]
        if self.classify.linear.in_features != 1280 or self.classify.linear.out_features != 3:
            raise ValueError('Expected the current three-class nano parent')
        self.train_last_blocks = int(train_last_blocks)
        self.tap_index = int(tap_index)
        self.decoder_channels = int(decoder_channels)
        self.embedding_dim = int(embedding_dim)
        if not 0 <= self.train_last_blocks <= len(self.backbone):
            raise ValueError('Invalid train_last_blocks')
        if not 0 <= self.tap_index < len(self.backbone):
            raise ValueError('Invalid early tap')
        if self.decoder_channels % 4:
            raise ValueError('decoder_channels must be divisible by4')
        if head_dropout_p is not None:
            self.classify.drop.p = float(head_dropout_p)
        self.head_dropout_p = float(self.classify.drop.p)
        sample = next(self.backbone.parameters())
        with torch.no_grad():
            early, late = self._features(torch.zeros(1, 3, 224, 224, device=sample.device))
        if tuple(early.shape[-2:]) != (28, 28) or tuple(late.shape[-2:]) != (7, 7):
            raise ValueError('This pilot requires28x28 early and7x7 late features at224')
        early_channels, late_channels = int(early.shape[1]), int(late.shape[1])
        axis = torch.linspace(0., 1., 28)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('coordinate_grid', torch.stack([xx, yy])[None])
        self.geometry_decoder = nn.Sequential(
            nn.Conv2d(early_channels + late_channels + 2, self.decoder_channels, 1, bias=False),
            nn.GroupNorm(4, self.decoder_channels), nn.SiLU(),
            nn.Conv2d(self.decoder_channels, self.decoder_channels, 3, padding=1,
                      groups=self.decoder_channels, bias=False),
            nn.Conv2d(self.decoder_channels, self.decoder_channels, 1, bias=False),
            nn.GroupNorm(4, self.decoder_channels), nn.SiLU(),
        )
        self.heatmap_head = nn.Conv2d(self.decoder_channels, len(HEATMAP_NAMES), 1)
        # Four-by-four bins retain which side of the queried crop features occupy.
        self.geometry_pool = nn.AvgPool2d(kernel_size=7, stride=7)
        geometry_dim = self.decoder_channels * 16 + len(HEATMAP_NAMES) * 2
        self.geometry_embedding = nn.Sequential(nn.LayerNorm(geometry_dim),
            nn.Linear(geometry_dim, self.embedding_dim), nn.SiLU())
        self.count_head = nn.Linear(self.embedding_dim, 3)
        self.visibility_head = nn.Linear(self.embedding_dim, 2 * len(VISIBILITY_NAMES))
        self.ownership_head = nn.Linear(self.embedding_dim, 2 * len(OWNERSHIP_NAMES))
        residual_dim = self.embedding_dim + 3 + 2 * len(VISIBILITY_NAMES) + 2 * len(OWNERSHIP_NAMES)
        self.class_residual = nn.Linear(residual_dim, 3)
        nn.init.zeros_(self.class_residual.weight)
        nn.init.zeros_(self.class_residual.bias)
        self.configure_trainability()
        self.eval()

    def configure_trainability(self):
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        frozen_until = len(self.backbone) - self.train_last_blocks
        for block in self.backbone[:frozen_until]:
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        frozen_until = len(self.backbone) - self.train_last_blocks
        for block in self.backbone[:frozen_until]:
            block.eval()
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def _features(self, images):
        early = None
        x = images
        for index, block in enumerate(self.backbone):
            x = block(x)
            if index == self.tap_index:
                early = x
        return early, x

    def _forward(self, images):
        if not torch.jit.is_tracing() and tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError('Expected NCHW RGB/stretch224 input')
        early, late = self._features(images)
        parent_features = self.classify.conv(late)
        parent_embedding = self.classify.pool(parent_features).flatten(1)
        parent_logits = self.classify.linear(self.classify.drop(parent_embedding))
        upsampled = F.interpolate(late, size=(28, 28), mode='bilinear', align_corners=False)
        grid = self.coordinate_grid.expand(images.shape[0], -1, -1, -1)
        dense = self.geometry_decoder(torch.cat([early, upsampled, grid], dim=1))
        heatmaps = self.heatmap_head(dense)
        spatial_probabilities = heatmaps.flatten(2).softmax(dim=-1)
        grid_flat = self.coordinate_grid.flatten(2).transpose(1, 2)
        coordinates = torch.matmul(spatial_probabilities, grid_flat)
        geometry = self.geometry_embedding(torch.cat([
            self.geometry_pool(dense).flatten(1), coordinates.flatten(1)], dim=1))
        count = self.count_head(geometry)
        visibility = self.visibility_head(geometry).reshape(-1, 2, len(VISIBILITY_NAMES))
        ownership = self.ownership_head(geometry).reshape(-1, 2, len(OWNERSHIP_NAMES))
        # Predicted quantities, never annotation truth, participate at inference.
        conditioned = torch.cat([geometry, count.softmax(-1),
                                 visibility.softmax(-1).flatten(1),
                                 ownership.softmax(-1).flatten(1)], dim=1)
        residual = self.class_residual(conditioned)
        return parent_logits + residual, dict(
            parent_logits=parent_logits, residual_logits=residual,
            heatmap_logits=heatmaps, coordinates=coordinates,
            count_logits=count, visibility_logits=visibility,
            ownership_logits=ownership, geometry_embedding=geometry)

    def forward(self, images, return_aux=False):
        logits, auxiliary = self._forward(images)
        return (logits, auxiliary) if return_aux else logits.softmax(-1)

    def parameter_groups(self, lr=3e-6, aux_lr=None):
        """Allow a faster LR for fresh auxiliary heads without changing parent LR."""
        aux_lr = lr if aux_lr is None else aux_lr
        parent, geometry = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (parent if name.startswith(('backbone.', 'classify.')) else geometry).append(parameter)
        return [dict(params=parent, lr=float(lr), group_name='parent_last_blocks_and_classifier'),
                dict(params=geometry, lr=float(aux_lr), group_name='geometry_branch')]

    def frozen_state(self):
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if name in buffers or (name in params and not params[name].requires_grad)}

    freeze_state = frozen_state

    def frozen_state_hash(self):
        return state_sha(self.frozen_state())

    def assert_frozen(self, reference):
        actual = self.frozen_state()
        if actual.keys() != reference.keys():
            raise ValueError('Frozen state policy/keys changed')
        changed = [name for name in actual if not torch.equal(actual[name], reference[name])]
        if changed:
            raise ValueError('Frozen early backbone/BN state changed: ' + str(changed))

    def configuration(self):
        return dict(train_last_blocks=self.train_last_blocks, tap_index=self.tap_index,
                    decoder_channels=self.decoder_channels, embedding_dim=self.embedding_dim,
                    head_dropout_p=self.head_dropout_p)


def from_parent(path, **configuration):
    return GrooveOwnershipClassifier(YOLO(str(path)).model.float().cpu().eval(), **configuration)


def save_model(path, model, **metadata):
    torch.save(dict(format=FORMAT, parent_yaml=model.parent_yaml, labels=LABELS,
                    configuration=model.configuration(),
                    state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    metadata=metadata), path)


def load_model(path, device='cpu'):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value.get('format') != FORMAT or value.get('labels') != LABELS:
        raise ValueError('Wrong groove ownership artifact')
    parent = ClassificationModel(cfg=value['parent_yaml'], nc=3, verbose=False)
    parent.names = dict(enumerate(LABELS))
    model = GrooveOwnershipClassifier(parent, **value['configuration'])
    model.load_state_dict(value['state_dict'], strict=True)
    return model.to(device).eval()


class InferenceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images)


def export_model(model, path, batch=2, opset=17):
    parameter = next(model.parameters())
    model.eval()
    sample = torch.zeros(batch, 3, 224, 224, device=parameter.device, dtype=parameter.dtype)
    torch.onnx.export(InferenceWrapper(model).eval(), sample, str(path),
        input_names=['images'], output_names=['probabilities'],
        dynamic_axes={'images': {0: 'batch'}, 'probabilities': {0: 'batch'}},
        opset_version=opset, do_constant_folding=True)


def gaussian_heatmaps(xy, size=28, sigma=1.25):
    """Normalized[0,1] Bx6x2 points to targets; caller applies explicit masks.

    Unknown/estimated points must have loss weight0. This helper deliberately
    never infers validity/visibility from coordinates or classifier labels.
    """
    axis = torch.arange(size, device=xy.device, dtype=xy.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    center = xy * (size - 1)
    dx = xx[None, None] - center[..., 0, None, None]
    dy = yy[None, None] - center[..., 1, None, None]
    return torch.exp(-(dx.square() + dy.square()) / (2 * sigma * sigma))


def smoke_test(parent_path, output_dir=None):
    """One random-tensor optimizer step; no dataset, receipt or full fit is used."""
    torch.set_num_threads(4)
    torch.manual_seed(1103)
    parent = YOLO(str(parent_path)).model.float().cpu().eval()
    model = GrooveOwnershipClassifier(parent).eval()
    images = torch.rand(2, 3, 224, 224)
    with torch.inference_mode():
        old = parent(images)
        old = old[0] if isinstance(old, (tuple, list)) else old
        new = model(images)
        logits, auxiliary = model(images, return_aux=True)
    difference = float((old - new).abs().max())
    assert difference == 0., difference
    assert torch.count_nonzero(auxiliary['residual_logits']) == 0
    assert auxiliary['heatmap_logits'].shape == (2, 6, 28, 28)
    assert auxiliary['coordinates'].shape == (2, 6, 2)
    assert auxiliary['visibility_logits'].shape == (2, 2, 4)
    assert auxiliary['ownership_logits'].shape == (2, 2, 3)
    frozen = model.frozen_state()
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.train()
    assert all(not m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
    for index, block in enumerate(model.backbone):
        assert all(p.requires_grad == (index >= 8) for p in block.parameters())
    optimizer = torch.optim.AdamW(model.parameter_groups(lr=3e-6, aux_lr=3e-4), weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits, auxiliary = model(images, return_aux=True)
    known = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0]], dtype=torch.float32)
    targets = gaussian_heatmaps(torch.rand(2, 6, 2))
    heatmap_loss = (F.binary_cross_entropy_with_logits(auxiliary['heatmap_logits'], targets,
                    reduction='none') * known[..., None, None]).sum() / (known.sum() * 28 * 28)
    loss = (F.cross_entropy(logits, torch.tensor([0, 2])) + heatmap_loss
            + F.cross_entropy(auxiliary['count_logits'], torch.tensor([2, 1]))
            + F.cross_entropy(auxiliary['visibility_logits'].flatten(0, 1), torch.tensor([0, 0, 1, 2]))
            + F.cross_entropy(auxiliary['ownership_logits'].flatten(0, 1), torch.tensor([0, 0, 0, 1])))
    loss.backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
        elif parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    gradient_groups = {}
    for prefix in ['backbone.8.', 'backbone.9.', 'classify.', 'geometry_decoder.', 'heatmap_head.', 'count_head.', 'visibility_head.', 'ownership_head.', 'class_residual.']:
        norm = sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
        assert norm > 0., (prefix, norm)
        gradient_groups[prefix] = norm
    optimizer.step()
    model.assert_frozen(frozen)
    changed = [n for n,p in model.named_parameters() if n in before and not torch.equal(p.detach(), before[n])]
    assert changed
    model.eval()
    result = dict(epoch0_parent_max_probability_error=difference, epoch0_parent_exact=True,
                  train_last_backbone_blocks=[8, 9], batchnorm_running_statistics_frozen=True,
                  frozen_state_sha256=model.frozen_state_hash(),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  geometry_parameters=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('backbone.', 'classify.'))),
                  gradient_groups=gradient_groups, changed_trainable_tensors=len(changed),
                  random_tensor_steps=1, full_fits=0)
    if output_dir is not None:
        import onnxruntime as ort
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir/'one_step_smoke.pt'
        save_model(checkpoint, model, smoke_only=True, random_tensor_steps=1)
        restored = load_model(checkpoint)
        with torch.inference_mode():
            expected = model(images)
            actual = restored(images)
        assert torch.equal(expected, actual)
        export_path = output_dir/'one_step_smoke.onnx'
        export_model(restored, export_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(export_path), sess_options=options, providers=['CPUExecutionProvider'])
        maximum = 0.
        for batch in [1, 2]:
            x = images[:batch]
            with torch.inference_mode():
                a = restored(x).numpy()
            b = session.run(None, {'images': x.numpy()})[0]
            maximum = max(maximum, float(np.abs(a-b).max()))
            assert np.array_equal(a.argmax(1), b.argmax(1))
        assert maximum < 1e-4, maximum
        result.update(save_load_exact=True, onnx_dynamic_batches_verified=[1, 2],
                      onnx_max_probability_error=maximum, onnx_argmax_matches=True)
    return result


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-parent', type=Path, required=True)
    parser.add_argument('--smoke-output', type=Path)
    args = parser.parse_args()
    print(json.dumps(smoke_test(args.smoke_parent, args.smoke_output), indent=2))
