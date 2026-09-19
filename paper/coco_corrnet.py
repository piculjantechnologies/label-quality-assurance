"""CorrNet: the paper's two-branch net with correlation fusion (README
item 4), the released architecture.

The paper architecture global-average-pools each trunk before the branches
meet, so no layer can compare a box with the object it annotates.

CorrNet is the smallest addition that makes the comparison. The trunks'
final maps are already aligned cell-for-cell (both stride-32 on the same
letterboxed 640 input), so an element-wise product per cell, pooled, gives a
512-d co-occurrence signal: does image-feature k appear where label-feature
k claims something? The first fc layer widens 1024 -> 1536 to accept it,
and the optional corr_masked / corr_max blocks (on L2-normalised features
with corr_cos) add 512 columns each; the released model enables all
three, so its fc layers read 2560 features.

The released model also initialises the image trunk from ImageNet with
the input normalised inside forward and its BatchNorm statistics frozen
(pretrained_img, input_norm, freeze_image_bn; README item 5), and carries
the raster head (raster_head; README item 9): a verification head that
reads the label raster itself at stride 8 next to the image trunk's
stride-8 and stride-16 features, scores every cell the label draws on,
and adds the claimed cells' worst, mean and count to the good logit
through a zero-initialised layer, so a model starts as exactly the
CorrNet function without it. GhostBatchNorm / ghostify (README item 8)
compute BatchNorm statistics per group of samples in a larger batch.

Warm start (a research option, train_coco_corr.py --warm) maps a paper-net
checkpoint exactly: trunks and fc tail copy over, the fc entry layer copies
into its first 1024 columns and zeroes the 512 correlation columns -- so at
initialisation the model computes exactly the source checkpoint's function,
and training grows the correlation term from there.
"""

import torch
from torch import nn
from torch.nn import functional as F

from neural_network import Net  # paper architecture, for the trunk layout


class CorrNet(nn.Module):
    """Optional sharpened-correlation blocks:

    corr_masked -- a second 512-d correlation pooled only over cells the
    label's box borders pass through, so one box in a dense scene is not
    diluted by the empty rest of the grid.
    corr_max -- per-channel max over cells, letting a single strongly
    (dis)agreeing cell speak regardless of scene density.
    corr_cos -- the added blocks correlate L2-normalised features, so
    high-magnitude activations elsewhere cannot dominate the agreement
    signal. The plain global-mean corr is never touched: extended
    models warm-started from a plain CorrNet checkpoint via adopt_corr
    start as exactly that checkpoint's function (added columns zeroed).
    The released model enables corr_masked, corr_max and corr_cos;
    corr16 / corr8 / corr4 are research options it does not use.

    The enabled blocks are recorded in a corr_cfg buffer inside the state
    dict (absent on plain models, which read as all flags off);
    from_checkpoint reads it back, so loaders need no flag plumbing.

    pretrained_img initialises the image trunk from ImageNet weights
    (torchvision IMAGENET1K_V1) instead of scratch. Initialisation only:
    the module layout is the same, and nothing is recorded in corr_cfg.

    raster_head adds the per-cell verification head (see __init__); a
    checkpoint that carries it is recognised by its rh_ layers.

    input_norm converts the pipeline's raw 0-255 BGR image to RGB with the
    ImageNet statistics inside forward, so a pretrained image trunk keeps
    its weights as published (without it, pretrained_img folds the
    convention into conv1 instead, a research option). It is recorded as
    the in_mean / in_std buffers, which from_checkpoint recognises.
    freeze_image_bn keeps the image trunk's BatchNorm layers in eval mode
    during training, so their statistics stay the pretrained ones while
    the weights fine-tune; it acts in training only and is not recorded.
    The released checkpoint is trained with pretrained_img, input_norm and
    freeze_image_bn (README item 5).
    """

    def __init__(self, device=None, corr_masked=False, corr_max=False,
                 corr_cos=False, corr16=False, corr8=False, corr4=False,
                 pretrained_img=False, raster_head=False, input_norm=False,
                 freeze_image_bn=False):
        super().__init__()
        base = Net()
        self.model_1 = base.model_1          # image trunk
        self.input_norm = input_norm
        self.freeze_image_bn = freeze_image_bn
        if input_norm:
            self.register_buffer("in_mean", torch.tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255.0)
            self.register_buffer("in_std", torch.tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255.0)
        if pretrained_img:
            from torchvision.models import ResNet18_Weights, resnet18
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            if not input_norm:
                self._fold_input_convention(m)
            self.model_1 = nn.Sequential(*list(m.children())[:-2])
        self.model_2 = base.model_2          # label-raster trunk
        self.corr_masked = corr_masked
        self.corr_max = corr_max
        self.corr_cos = corr_cos
        self.corr16 = corr16
        self.corr8 = corr8
        self.corr4 = corr4
        width = (1536 + 512 * (corr_masked + corr_max)
                 + 256 * corr16 + 128 * corr8 + 64 * corr4)
        flags = [corr_masked, corr_max, corr_cos, corr16, corr8, corr4]
        if any(flags):
            self.register_buffer("corr_cfg",
                                 torch.tensor(flags, dtype=torch.uint8))
        self.raster_head = raster_head
        if raster_head:
            # The raster head (README item 9). The label is judged cell by
            # cell on the stride-8 grid: each cell sees the image
            # trunk's stride-8 and stride-16 features
            # next to the raster itself -- which classes' outlines cross the
            # cell (each class plane max-pooled 8 x 8) and where inside the
            # cell they run (the class-agnostic outline, pixel-unshuffled
            # into 64 channels, so a shift of a pixel is still visible).
            # Dilated convolutions give each cell the context around a box
            # edge. Per cell it scores the support for every class and the
            # class the image actually shows there (80 = nothing); the
            # raster's own class planes pick the claimed classes' support.
            # The verdict mixes a smooth minimum, the mean and the count of
            # the claimed cells' support into the good logit through
            # box_mix, zero-initialised.
            def block(cin, cout, dil):
                return [nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil),
                        nn.GroupNorm(16, cout), nn.ReLU()]
            self.rh_body = nn.Sequential(
                *block(128 + 256 + 80 + 64, 256, 1), *block(256, 256, 2),
                *block(256, 256, 4), *block(256, 128, 8))
            self.rh_support = nn.Conv2d(128, 80, 1)
            self.rh_cls = nn.Conv2d(128, 81, 1)
            self.box_mix = nn.Linear(3, 1)
            nn.init.zeros_(self.box_mix.weight)
            nn.init.zeros_(self.box_mix.bias)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 2))
        if device is not None:
            self.to(device)

    @staticmethod
    def _fold_input_convention(m):
        """Reparameterise conv1/bn1 so the ImageNet function is computed
        directly on the pipeline's raw 0-255 BGR input.

        get_sample feeds 0-255 BGR; the ImageNet weights expect /255 RGB
        normalised by (mean, std). The two conventions differ by an
        affine, per-channel change of input variables, which the first
        conv absorbs exactly everywhere except the 2-px output border
        (where conv1's zero padding stands for a different constant under
        the two conventions): rescale each RGB kernel by 1/(255*std),
        reorder kernels to BGR, and shift bn1's running mean by the
        constant the dropped mean-subtraction contributed.
        """
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        with torch.no_grad():
            w = m.conv1.weight / (255.0 * std.view(1, 3, 1, 1))
            m.bn1.running_mean.add_(
                (w * (255.0 * mean).view(1, 3, 1, 1)).sum((1, 2, 3)))
            m.conv1.weight.copy_(w.flip(1))

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_image_bn:
            for m in self.model_1.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def _product(self, a, b):
        if self.corr_cos:
            return F.normalize(a, dim=1) * F.normalize(b, dim=1)
        return a * b

    @staticmethod
    def _box_masked(prod, cats):
        """Pool the agreement over the cells the label's box borders touch.

        Dilution is the whole point of pooling finer: a stride-8 grid has
        6400 cells to stride-32's 400, so a global mean would bury one
        box's borders 16x deeper. The mask is the label raster reduced to
        the feature map's own resolution, so it lands cell-for-cell.
        """
        stride = cats.shape[-1] // prod.shape[-1]
        mask = (F.max_pool2d(cats.amax(1, keepdim=True), stride)
                > 0).to(prod.dtype)
        denom = mask.sum(dim=(2, 3)).clamp(min=1.0)
        return (prod * mask).sum(dim=(2, 3)) / denom

    def forward(self, cats, background, aux=False):
        """aux=True (a raster_head model) also returns the head's per-cell
        support (B, 80, g, g) and class logits (B, 81, g, g), which its
        training losses read."""
        if self.input_norm:
            # letterbox padding (0) maps to -mean/std, as it does when a
            # black-padded image is normalised
            background = (background.flip(1) - self.in_mean) / self.in_std
        if self.corr16 or self.corr8 or self.corr4 or self.raster_head:
            # the paper's trunk is an nn.Sequential of resnet18's children:
            # [:5] ends after layer1 (stride 4), [5] is layer2 (stride 8),
            # [6] is layer3 (stride 16), [7] is layer4 (stride 32). Running
            # it staged yields the same tensors the single call would, and
            # adds no parameters.
            i4_img = self.model_1[:5](background)
            i8_img = self.model_1[5](i4_img)
            i16_img = self.model_1[6](i8_img)
            f_img = self.model_1[7](i16_img)
            i4_lbl = self.model_2[:5](cats)
            i8_lbl = self.model_2[5](i4_lbl)
            i16_lbl = self.model_2[6](i8_lbl)
            f_lbl = self.model_2[7](i16_lbl)
        else:
            f_img = self.model_1(background)             # B x 512 x 20 x 20
            f_lbl = self.model_2(cats)
        p_img = F.adaptive_avg_pool2d(f_img, 1).flatten(1)
        p_lbl = F.adaptive_avg_pool2d(f_lbl, 1).flatten(1)
        corr = (f_img * f_lbl).mean(dim=(2, 3))          # per-cell agreement
        parts = [p_img, p_lbl, corr]
        if self.corr_masked or self.corr_max:
            prod = self._product(f_img, f_lbl)
            if self.corr_masked:
                parts.append(self._box_masked(prod, cats))
            if self.corr_max:
                parts.append(prod.amax(dim=(2, 3)))
        # finer taps are always box-masked: see _box_masked
        if self.corr16:
            parts.append(self._box_masked(self._product(i16_img, i16_lbl),
                                          cats))
        if self.corr8:
            parts.append(self._box_masked(self._product(i8_img, i8_lbl),
                                          cats))
        if self.corr4:
            parts.append(self._box_masked(self._product(i4_img, i4_lbl),
                                          cats))
        if self.raster_head:
            return self._raster(self.fc(torch.cat(parts, dim=1)), cats,
                                i8_img, i16_img, aux)
        return self.fc(torch.cat(parts, dim=1))

    def _raster(self, logits, cats, i8, i16, aux=False):
        """The raster head (see __init__): per-cell class support and
        class, and the claimed cells' support mixed into the good logit."""
        k = cats.shape[-1] // i8.shape[-1]
        r_cls = F.max_pool2d(cats, k)
        r_geo = F.pixel_unshuffle(cats.amax(1, keepdim=True), k)
        h = self.rh_body(torch.cat(
            [i8, F.interpolate(i16, size=i8.shape[-2:], mode="bilinear",
                               align_corners=False),
             r_cls.to(i8.dtype), r_geo.to(i8.dtype)], 1))
        sup = self.rh_support(h).float()
        claimed = r_cls > 0
        cnt = claimed.flatten(1).sum(1).float()
        has = cnt > 0
        # smooth minimum -logsumexp(-v) over the claimed cells; unclaimed
        # cells enter at a finite -1e4 (not -inf) so a label with no
        # claimed cell backpropagates zeros rather than NaN
        smin = -torch.logsumexp(
            torch.where(claimed, -sup, torch.full_like(sup, -1e4))
            .flatten(1), 1)
        smin = torch.where(has, smin, torch.zeros_like(smin))
        mean = torch.where(claimed, sup, torch.zeros_like(sup)) \
            .flatten(1).sum(1) / cnt.clamp(min=1)
        delta = self.box_mix(torch.stack(
            [smin, mean, torch.log1p(cnt)], 1)).squeeze(1)
        out = logits + torch.stack([torch.zeros_like(delta), delta],
                                   1).to(logits.dtype)
        return (out, sup, self.rh_cls(h)) if aux else out

    N_FLAGS = 6

    @classmethod
    def _cfg_of(cls, sd):
        """The flag vector a state dict records, padded to the current
        length; entries missing from a shorter vector are off."""
        cfg = sd.get("corr_cfg", torch.zeros(cls.N_FLAGS, dtype=torch.uint8))
        if cfg.numel() < cls.N_FLAGS:
            cfg = torch.cat([cfg, torch.zeros(cls.N_FLAGS - cfg.numel(),
                                              dtype=cfg.dtype)])
        return cfg

    @classmethod
    def from_checkpoint(cls, ckpt_path, device=None):
        """Construct with the flags recorded in the checkpoint (and the
        raster head if it has rh_ layers), load it."""
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True) \
            if not isinstance(ckpt_path, dict) else ckpt_path
        cfg = cls._cfg_of(sd)
        model = cls(device, corr_masked=bool(cfg[0]), corr_max=bool(cfg[1]),
                    corr_cos=bool(cfg[2]), corr16=bool(cfg[3]),
                    corr8=bool(cfg[4]), corr4=bool(cfg[5]),
                    raster_head=any(k.startswith("rh_") for k in sd),
                    input_norm="in_mean" in sd)
        sd = dict(sd)
        if "corr_cfg" in sd:
            sd["corr_cfg"] = cfg
        model.load_state_dict(sd if device is None else
                              {k: v.to(device) for k, v in sd.items()})
        return model

    def warm_start(self, ckpt_path):
        """Adopt a paper-net checkpoint; start as exactly that function."""
        src = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        own = self.state_dict()
        for k, v in src.items():
            if k == "fc.1.weight":
                own[k][:, :1024] = v
                own[k][:, 1024:] = 0.0
            else:
                own[k].copy_(v)
        self.load_state_dict(own)
        return self

    def adopt_corr(self, src):
        """Adopt a narrower CorrNet checkpoint into an extended model; start
        as exactly that function, the added blocks' fc columns zeroed so
        training grows them from nothing.

        Blocks concatenate in a fixed order (p_img, p_lbl, corr, masked,
        max, corr16, corr8), so the source's columns are a prefix of this
        model's only if every block the source had is still enabled here --
        enabling corr16 while dropping corr_masked would silently misalign,
        so that is refused rather than trusted.
        """
        if not isinstance(src, dict):
            src = torch.load(src, map_location="cpu", weights_only=True)
        src_cfg, own_cfg = self._cfg_of(src), self._cfg_of(self.state_dict())
        # cos (index 2) changes no widths, so it is free to differ
        for i in (0, 1, 3, 4, 5):
            if src_cfg[i] and not own_cfg[i]:
                raise ValueError(
                    f"source enables corr flag {i} that this model does not; "
                    "its fc columns would not be a prefix of this model's")
        own = self.state_dict()
        for k, v in src.items():
            if k == "corr_cfg":
                continue                      # this model records its own
            if k == "fc.1.weight" and v.shape[1] != own[k].shape[1]:
                own[k][:, :v.shape[1]] = v
                own[k][:, v.shape[1]:] = 0.0
            else:
                own[k].copy_(v)
        self.load_state_dict(own)
        return self


class _GhostBNMixin:
    """BatchNorm whose training statistics are computed per virtual
    sub-batch of `virtual_bs` samples, reproducing the gradient-
    accumulation recipe's BN semantics (micro-batches of 8) inside one
    large physical batch (README item 8). Eval behaviour is the standard
    BatchNorm, and so is a layer in eval mode during training (the image
    trunk's under freeze_image_bn)."""

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
