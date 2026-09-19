"""Unified model: spatial fusion-before-pooling.

Dataset-agnostic; class count, input resolution and fusion grid come from
`dataset.py` (selected by QA_DATASET=coco|voc). Inputs are ImageNet-normalised
RGB images and the qa_data label planes (filled interior, border,
diagonals). Both trunks keep their spatial maps until after they meet;
the fused grid is pooled by average, maximum and labelled-region mean,
and the head additionally sees a per-class claimed-position agreement
vector (class_corr, supervised by the auxiliary box-class loss) and its
lower-quartile worst-box variant. A per-box verification head scores
every claimed box alone (support logit and signed band margin) and feeds
the sample head their minimum and mean. GhostBatchNorm / ghostify supply
per-sub-batch BatchNorm statistics for the one-physical-batch recipe.
"""

import dataset as _dataset


import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (resnet18, ResNet18_Weights,
                                resnet50, ResNet50_Weights)

STAGES = ("stem", "layer1", "layer2", "layer3", "layer4")


class _GhostBNMixin:
    """BatchNorm whose training statistics are computed per virtual
    sub-batch of `virtual_bs` samples, reproducing the gradient-
    accumulation recipe's BN semantics (micro-batches of 8) inside one
    large physical batch. A batch whose size is not a multiple of
    `virtual_bs` uses whole-batch statistics. Eval behaviour is the
    standard BatchNorm."""

    def forward(self, x):
        vb = self.virtual_bs
        if not self.training or x.shape[0] <= vb \
                or x.shape[0] % vb != 0:
            return super().forward(x)
        return torch.cat([super(_GhostBNMixin, self).forward(c)
                          for c in x.chunk(x.shape[0] // vb, dim=0)], 0)


class GhostBatchNorm2d(_GhostBNMixin, nn.BatchNorm2d):
    def __init__(self, num_features, virtual_bs=8, **kw):
        super().__init__(num_features, **kw)
        self.virtual_bs = virtual_bs


class GhostBatchNorm1d(_GhostBNMixin, nn.BatchNorm1d):
    def __init__(self, num_features, virtual_bs=8, **kw):
        super().__init__(num_features, **kw)
        self.virtual_bs = virtual_bs


def ghostify(module, virtual_bs=8):
    """Replace every BatchNorm1d/2d in `module` with its Ghost variant,
    keeping weights, statistics, and flags."""
    for name, child in module.named_children():
        cls = None
        if isinstance(child, nn.BatchNorm2d) and \
                not isinstance(child, GhostBatchNorm2d):
            cls = GhostBatchNorm2d
        elif isinstance(child, nn.BatchNorm1d) and \
                not isinstance(child, GhostBatchNorm1d):
            cls = GhostBatchNorm1d
        if cls is not None:
            g = cls(child.num_features, virtual_bs,
                    eps=child.eps, momentum=child.momentum,
                    affine=child.affine,
                    track_running_stats=child.track_running_stats)
            g.load_state_dict(child.state_dict())
            g.training = child.training
            for p_g, p_c in zip(g.parameters(), child.parameters()):
                p_g.requires_grad = p_c.requires_grad
            setattr(module, name, g)
        else:
            ghostify(child, virtual_bs)
    return module
WORST_Q = 0.25
# Which trunk levels feed the fusion grid, by its stride at the input
# resolution: a stride-8 grid needs all three maps, stride-16 the two
# coarser ones, stride-32 the last alone.
LEVELS_FOR_STRIDE = {8: (0, 1, 2), 16: (1, 2), 32: (2,)}
FREEZE_CHOICES = ("none", "stem", "layer1", "layer2", "layer3", "layer4")


class Trunk(nn.Module):
    """ResNet exposing its stride-8/16/32 feature maps."""

    ARCH_CHANNELS = {"resnet18": (128, 256, 512),
                     "resnet50": (512, 1024, 2048)}

    def __init__(self, in_channels=3, pretrained=False, arch="resnet18"):
        super().__init__()
        # The image trunk consumes qa_data's ImageNet-normalised RGB
        # directly, so pretrained weights load unchanged.
        if arch == "resnet50":
            r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2
                         if pretrained else None)
        else:
            r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1
                         if pretrained else None)
        self.channels = self.ARCH_CHANNELS[arch]
        if in_channels != 3:
            r.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.layer3, self.layer4 = r.layer3, r.layer4
        self.frozen = []

    def freeze_through(self, stage):
        """Freeze every stage up to and including `stage` (weights and BN)."""
        if stage == "none":
            return
        cutoff = STAGES.index(stage)
        for name in STAGES[:cutoff + 1]:
            module = getattr(self, name)
            for p in module.parameters():
                p.requires_grad = False
            self.frozen.append(module)

    def train(self, mode=True):
        super().train(mode)
        for module in self.frozen:      # keep frozen BN statistics fixed
            module.eval()
        return self

    def forward(self, x):
        c2 = self.layer2(self.layer1(self.stem(x)))   # channels[0], stride 8
        c3 = self.layer3(c2)                          # channels[1], stride 16
        return c2, c3, self.layer4(c3)                # channels[2], stride 32
        # (channel counts per trunk in ARCH_CHANNELS)


class FusionNet(nn.Module):
    def __init__(self, num_classes=None, dim=256, fusion_res=None,
                 input_res=None, freeze_through="layer2", dropout=0.2,
                 region_pool=True, class_corr=True, worst_pool=True,
                 count_head=False, per_box=False, image_arch="resnet18",
                 device=None,
                 pretrained_image=True):
        super().__init__()
        num_classes = num_classes or _dataset.NUM_CLASSES
        fusion_res = fusion_res or _dataset.FUSION_RES
        input_res = input_res or _dataset.RES
        self.region_pool = region_pool
        self.class_corr = class_corr
        self.worst_pool = worst_pool and class_corr
        self.count_head = count_head
        self.per_box = per_box
        stride = input_res // fusion_res
        if stride not in LEVELS_FOR_STRIDE or stride * fusion_res != input_res:
            raise ValueError(f"fusion_res {fusion_res} at input {input_res} "
                             f"gives stride {stride}; need one of "
                             f"{sorted(LEVELS_FOR_STRIDE)}")
        self.fusion_res = fusion_res
        self.input_res = input_res
        self.levels = LEVELS_FOR_STRIDE[stride]

        # The label trunk stays resnet18: it reads synthetic planes with
        # no pretraining to inherit, so extra capacity buys nothing there.
        self.image_trunk = Trunk(3, pretrained=pretrained_image, arch=image_arch)
        self.image_trunk.freeze_through(freeze_through)
        self.label_trunk = Trunk(num_classes)

        # Bare state dicts self-describe (from_checkpoint reads this).
        self.register_buffer("fusion_cfg", torch.tensor(
            [fusion_res, FREEZE_CHOICES.index(freeze_through), input_res],
            dtype=torch.int32))

        def reduce(trunk):
            in_ch = sum(trunk.channels[i] for i in self.levels)
            return nn.Sequential(
                nn.Conv2d(in_ch, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

        self.reduce_image = reduce(self.image_trunk)
        self.reduce_label = reduce(self.label_trunk)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True))

        if class_corr:
            # Per-class visual evidence map: supervised by the auxiliary
            # box-class loss (qa_train.py --aux_weight), consumed as
            # a per-class claimed-vs-seen agreement vector. This is the
            # mechanism for detecting class swaps: a swapped box shows
            # object evidence but not in its claimed class channel.
            self.class_proj = nn.Conv2d(dim, num_classes, kernel_size=1)
        if count_head:
            # Per-class object-density map over the image alone: its spatial
            # sum is a predicted per-class count, supervised against the
            # ground-truth counts (qa_train.py --count_weight). Gives
            # the head the label-vs-image bookkeeping signal that catches
            # spurious and missing boxes independent of local appearance.
            self.count_proj = nn.Conv2d(dim, num_classes, kernel_size=1)
        if per_box:
            # Per-box verification: each claimed box is scored alone from
            # RoI features over the fused grid, a 2x context window, and the
            # box's claimed-class evidence channel -- so one spurious box
            # among genuine same-class boxes cannot average away, the
            # failure mode of the per-class agreement pools. Outputs a
            # support logit and a signed band-margin regression per box
            # (supervised from qa_data.per_box_targets); the head sees the
            # per-sample minimum and mean of the support logits, the
            # differentiable form of the metric's "bad if ANY box is bad".
            if not class_corr:
                raise ValueError("per_box requires class_corr")
            self.box_reduce = nn.Sequential(
                nn.Conv2d(dim, 64, kernel_size=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.box_mlp = nn.Sequential(
                nn.Linear(64 * 9 * 2 + 9 + 5, 128), nn.BatchNorm1d(128),
                nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(128, 2))
        head_in = dim * (3 if region_pool else 2) \
            + (num_classes if class_corr else 0) \
            + (num_classes if self.worst_pool else 0) \
            + (num_classes if count_head else 0) \
            + (2 if per_box else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 2))
        if device is not None:
            self.to(device)

    def _pyramid(self, reduce, feats):
        sel = [feats[i] for i in self.levels]
        size = sel[0].shape[-2:]
        merged = [sel[0]] + [F.interpolate(f, size=size, mode="bilinear",
                                           align_corners=False)
                             for f in sel[1:]]
        return reduce(torch.cat(merged, 1))

    def forward(self, cats, background, boxes=None, return_class_map=False,
                return_aux=False):
        fi = self._pyramid(self.reduce_image, self.image_trunk(background))
        fl = self._pyramid(self.reduce_label, self.label_trunk(cats))
        z = self.fuse(torch.cat([fi, fl, fi * fl], 1))
        zf = z.flatten(2)
        pools = [zf.mean(-1), zf.amax(-1)]
        if self.region_pool:
            # A sparse correct label must not dilute into background
            # statistics: the labeled positions get their own pool, so a
            # single-box positive carries the same-strength agreement
            # signal as a dense one.
            mask = (cats.amax(1, keepdim=True) > 0).float()
            mask = F.adaptive_max_pool2d(mask, z.shape[-2:])
            denom = mask.sum((2, 3)).clamp_min(1.0)
            pools.append((z * mask).sum((2, 3)) / denom)
        class_map = None
        if self.class_corr:
            class_map = self.class_proj(fi)              # B, C, H, W
            claimed = (F.adaptive_max_pool2d(cats, class_map.shape[-2:])
                       > 0).float()                       # B, C, H, W
            denom_c = claimed.sum((2, 3)).clamp_min(1.0)
            # Mean class evidence over the positions where that class is
            # claimed; zero for unclaimed classes.
            pools.append((class_map * claimed).sum((2, 3)) / denom_c)
            if self.worst_pool:
                # Mean agreement lets one bad box among good same-class
                # boxes average away, but the label is bad if ANY box is
                # bad: the head also sees the low quartile of claimed-
                # position evidence, a robust worst-box signal.
                worst = class_map.float().masked_fill(
                    claimed == 0, float("nan"))
                worst = torch.nanquantile(worst.flatten(2), WORST_Q, dim=-1)
                pools.append(torch.nan_to_num(worst, nan=0.0).to(z.dtype))
        counts = None
        if self.count_head:
            density = F.softplus(self.count_proj(fi).float())
            counts = density.sum((2, 3))
            pools.append(torch.log1p(counts).to(z.dtype))
        box_out = None
        if self.per_box:
            if boxes is None:
                raise ValueError("per_box model needs boxes: (N, 6) rows "
                                 "[sample, x1, y1, x2, y2, class] in "
                                 "input-letterbox pixels")
            B = z.shape[0]
            if boxes.shape[0]:
                from torchvision.ops import roi_align
                rois = boxes[:, :5].to(z.dtype)
                scale = self.fusion_res / float(self.input_res)
                zr = self.box_reduce(z)
                inner = roi_align(zr, rois, output_size=3,
                                  spatial_scale=scale, aligned=True)
                cx = (rois[:, 1] + rois[:, 3]) / 2
                cy = (rois[:, 2] + rois[:, 4]) / 2
                w = rois[:, 3] - rois[:, 1]
                h = rois[:, 4] - rois[:, 2]
                ctx = torch.stack([rois[:, 0],
                                   (cx - w).clamp(0, self.input_res),
                                   (cy - h).clamp(0, self.input_res),
                                   (cx + w).clamp(0, self.input_res),
                                   (cy + h).clamp(0, self.input_res)], 1)
                outer = roi_align(zr, ctx, output_size=3,
                                  spatial_scale=scale, aligned=True)
                cm = roi_align(class_map.to(z.dtype), rois, output_size=3,
                               spatial_scale=scale, aligned=True)
                cm = cm[torch.arange(cm.shape[0], device=cm.device),
                        boxes[:, 5].long()]
                geom = torch.stack([
                    w / self.input_res, h / self.input_res,
                    torch.log((w + 1) / (h + 1)),
                    cx / self.input_res, cy / self.input_res], 1)
                box_out = self.box_mlp(torch.cat(
                    [inner.flatten(1), outer.flatten(1),
                     cm.flatten(1), geom.to(z.dtype)], 1))
                s = box_out[:, 0].float()
                bidx = boxes[:, 0].long()
                cnt = torch.zeros(B, device=s.device).index_add_(
                    0, bidx, torch.ones_like(s))
                mean = torch.zeros(B, device=s.device).index_add_(
                    0, bidx, s) / cnt.clamp_min(1.0)
                mn = torch.full((B,), float("inf"), device=s.device) \
                    .scatter_reduce(0, bidx, s, reduce="amin",
                                    include_self=True)
                mn = torch.where(cnt > 0, mn, torch.zeros_like(mn))
                pools.append(torch.stack([mn, mean], 1).to(z.dtype))
            else:
                box_out = z.new_zeros((0, 2))
                pools.append(z.new_zeros((B, 2)))
        out = self.head(torch.cat(pools, 1))
        if return_aux:
            return out, class_map, counts, box_out
        if return_class_map:
            return out, class_map
        return out

    def param_groups(self, lr, finetune_mult=0.1):
        """Trainable pretrained stages (image layer3/4) get a smaller step."""
        pretrained_ids = {id(p) for p in self.image_trunk.parameters()}
        pretrained, fresh = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (pretrained if id(p) in pretrained_ids else fresh).append(p)
        groups = [{"params": fresh, "lr": lr}]
        if pretrained:
            groups.append({"params": pretrained, "lr": lr * finetune_mult})
        return groups

    @classmethod
    def from_checkpoint(cls, ckpt_path, device=None):
        try:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError:
            # a resumable training checkpoint carries optimizer and RNG state
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        cfg = sd["fusion_cfg"]
        dim = sd["fuse.0.weight"].shape[0]
        nc = sd["label_trunk.stem.0.weight"].shape[1]
        class_corr = "class_proj.weight" in sd
        count_head = "count_proj.weight" in sd
        per_box = "box_mlp.0.weight" in sd
        # Bottleneck blocks (conv3) mark a ResNet-50 image trunk.
        image_arch = ("resnet50"
                      if "image_trunk.layer1.0.conv3.weight" in sd
                      else "resnet18")
        # The head width self-describes the parameter-free pools: strip the
        # per-class vectors owned by class_proj/count_proj and the per-box
        # pair, then whatever nc-sized remainder is not dim*2 or dim*3 is
        # the worst-box pool.
        base = sd["head.0.weight"].shape[1] \
            - nc * class_corr - nc * count_head - 2 * per_box
        worst_pool = class_corr and base not in (dim * 2, dim * 3)
        if worst_pool:
            base -= nc
        if base not in (dim * 2, dim * 3):
            raise ValueError(f"cannot infer pools from head width "
                             f"{sd['head.0.weight'].shape[1]}")
        model = cls(num_classes=nc, fusion_res=int(cfg[0]),
                    freeze_through=FREEZE_CHOICES[int(cfg[1])],
                    input_res=int(cfg[2]),
                    region_pool=base == dim * 3, class_corr=class_corr,
                    worst_pool=worst_pool, count_head=count_head,
                    per_box=per_box, image_arch=image_arch,
                    pretrained_image=False)  # the checkpoint supplies them
        model.load_state_dict(sd)
        if device is not None:
            model.to(device)
        model.eval()
        return model


def load_any(ckpt_path, device=None):
    """Load a unified-pipeline checkpoint."""
    return FusionNet.from_checkpoint(ckpt_path, device)
