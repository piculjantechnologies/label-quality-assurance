"""Label quality assurance networks: CorrNet (the released network) and
NetSmall (thesis Listings 8.1 and 8.2).

CorrNet is the network train.py builds and the release ships (README items
2, 4 and 5): two ResNet-18 branches run to layer4 -- the image branch
ImageNet-pretrained and fine-tuned, its BatchNorm statistics kept at the
pretrained values; the label branch randomly initialised with a 20-channel
first convolution -- and a correlation fusion of their aligned stride-32
maps feeds a fully connected classifier with two outputs (0 = bad label,
1 = good label); with raster_head, the per-cell head adds the claimed
cells' support to the good logit.

NetSmall is the network of the printed listings (README finding 1): two
ResNet-18 branches truncated after their first residual stage -- the image
branch ImageNet-pretrained and frozen, the label branch randomly
initialised with a 20-channel first convolution -- whose pooled features
are projected to 128 dimensions and fused with a cross-attention layer
that feeds a small fully connected classifier. As printed, the
cross-attention has a single key/value token, so its softmax is
identically 1 and the classifier output is a function of the label branch
alone -- the image cannot influence any verdict. corr=False reproduces the
listings-as-published network exactly. With corr=True (NetSmall's default)
the aligned stage-1 maps of the two branches are compared position by
position: their element-wise product is pooled (mean and max over the
grid) and appended to the classifier input, and the image branch's
BatchNorm layers are kept in eval mode during training. The
cross-attention path itself stays image-blind in both variants (its
single-token softmax gives proj_1 zero gradient); in the correlation
variant the image enters the classifier through the product term and,
with raster_head, through the per-cell head. NetSmall can also fine-tune
its image layers (set_finetune_image). No NetSmall checkpoint ships;
load_model loads any of the three networks.

GhostBatchNorm1d/2d and ghostify compute training BatchNorm statistics per
group of samples (train.py --ghost_bn; the recipe uses groups of 8). """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torchvision.models import resnet18, ResNet18_Weights


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim1,
                                               num_heads=num_heads,
                                               batch_first=True)
        self.proj = nn.Linear(dim2, dim1)

    def forward(self, x1, x2):
        x2_proj = self.proj(x2).unsqueeze(1)
        x1 = x1.unsqueeze(1)
        attn_output, _ = self.attention(x1, x2_proj, x2_proj)
        return attn_output.squeeze(1)


class NetSmall(nn.Module):
    def __init__(self, num_classes=20, corr=True, pretrained=True,
                 raster_head=False):
        super().__init__()
        self.corr = corr
        self.raster_head = raster_head
        # set by set_finetune_image: the pretrained image layers train
        # instead of staying frozen
        self.finetune_image = False

        # pretrained=False skips the torchvision download when a checkpoint
        # is about to overwrite the trunk anyway (load_model).
        resnet_1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1
                            if pretrained else None)
        for param in resnet_1.parameters():
            param.requires_grad = False
        model_1 = nn.Sequential(*list(resnet_1.children())[:-5])

        model_2 = resnet18()
        model_2.conv1 = nn.Conv2d(num_classes, 64, kernel_size=7, stride=2,
                                  padding=3, bias=False)
        model_2 = nn.Sequential(*list(model_2.children())[:-5])

        self.model_1 = model_1
        self.model_2 = model_2

        self.adaptive_output_1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.adaptive_output_2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.proj_1 = nn.Linear(64, 128)
        self.proj_2 = nn.Linear(64, 128)

        self.cross_attention = CrossAttention(dim1=128, dim2=128, num_heads=32)

        if raster_head:
            # The label is judged cell by cell on the stride-8 grid (28 x 28
            # for 224-px inputs). Each cell sees semantic image features --
            # the pretrained ResNet-18's layer2 (stride 8) and layer3
            # (stride 16, upsampled), continued from the image trunk's
            # stage-1 map and frozen or fine-tuned with it -- next to the
            # label planes themselves: which classes are drawn in the cell
            # (each plane max-pooled 8 x 8) and where inside it they run (the
            # class-agnostic drawing, pixel-unshuffled into 64 channels, so
            # a one-pixel shift stays visible). Four 3 x 3 blocks with
            # dilations 1, 2, 4 and 8 widen each cell's view around a box
            # edge; per cell the head scores the support for every class
            # and the class the image shows there (num_classes = nothing).
            # The label's own planes pick the claimed classes' support, and
            # the verdict adds a smooth minimum and the mean of the claimed
            # cells' support and log(1 + their count) to the good logit
            # through box_mix, initialised at zero: the network starts as
            # exactly the fusion without the head.
            self.rh_trunk = nn.Sequential(resnet_1.layer2, resnet_1.layer3)
            def block(cin, cout, dil):
                return [nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil),
                        nn.GroupNorm(16, cout), nn.ReLU(inplace=True)]
            self.rh_body = nn.Sequential(
                *block(128 + 256 + num_classes + 64, 256, 1),
                *block(256, 256, 2), *block(256, 256, 4),
                *block(256, 128, 8))
            self.rh_support = nn.Conv2d(128, num_classes, 1)
            self.rh_cls = nn.Conv2d(128, num_classes + 1, 1)
            self.box_mix = nn.Linear(3, 1)
            nn.init.zeros_(self.box_mix.weight)
            nn.init.zeros_(self.box_mix.bias)
        fc_in = 64 * 2 + (128 if corr else 0)
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.corr and mode:
            # Keep the pretrained image layers' BatchNorm running
            # statistics frozen (their weights may be fine-tuned).
            self.model_1.eval()
            if self.raster_head:
                self.rh_trunk.eval()
        return self

    def set_finetune_image(self, on=True):
        """Unfreeze the pretrained image layers (the trunk and, with the
        raster head, its layer2/layer3) for fine-tuning; their BatchNorm
        statistics stay frozen (train keeps them in eval mode)."""
        self.finetune_image = on
        mods = [self.model_1] + ([self.rh_trunk] if self.raster_head else [])
        for m in mods:
            for p in m.parameters():
                p.requires_grad = on
        return [p for m in mods for p in m.parameters()]

    def _image(self, module, x):
        """A pretrained image module over a full batch. Frozen: no_grad in
        512-row chunks (identical output whenever the module is in eval
        mode, as in the correlation variant; bounded memory). Fine-tuned:
        activation checkpointing per 512-row chunk, so a large batch stores
        only each chunk's input."""
        if not self.finetune_image or not torch.is_grad_enabled():
            with torch.no_grad():
                return torch.cat([module(c) for c in x.split(512)], 0)
        return torch.cat([checkpoint(module, c, use_reentrant=False)
                          for c in x.split(512)], 0)

    def forward(self, img, label, aux=False):
        """aux=True (a raster_head model) also returns the head's per-cell
        support (B, num_classes, g, g) and class logits
        (B, num_classes + 1, g, g), which its training losses read."""
        # The image trunk is in eval mode in the correlation variant
        # (corr=True), so its forward is per-sample and chunking it
        # (self._image) is exactly equivalent while bounding the transient
        # activation memory of a large batch.
        map_1 = self._image(self.model_1, img)
        map_2 = self.model_2(label)

        output_1 = self.adaptive_output_1(map_1)
        output_2 = self.adaptive_output_2(map_2)

        output_1 = output_1.view(output_1.size(0), -1)
        output_2 = output_2.view(output_2.size(0), -1)

        output_1 = self.proj_1(output_1)
        output_2 = self.proj_2(output_2)

        attended_features = self.cross_attention(output_1, output_2)
        if not self.corr:
            return self.fc(attended_features)

        # Position-wise agreement between the aligned stage-1 maps: does
        # image evidence appear where the label planes claim something?
        product = map_1 * map_2
        corr = torch.cat([product.mean(dim=(2, 3)),
                          product.amax(dim=(2, 3))], dim=1)
        logits = self.fc(torch.cat([attended_features, corr], dim=1))
        if not self.raster_head:
            return logits
        return self._raster(logits, label, map_1, aux)

    def _raster(self, logits, label, map_1, aux=False):
        """The label-plane head (see __init__): per-cell class support and
        class, and the claimed cells' support mixed into the good logit."""
        f8 = self._image(self.rh_trunk[0], map_1)
        f16 = F.interpolate(self._image(self.rh_trunk[1], f8),
                            size=f8.shape[-2:], mode="bilinear",
                            align_corners=False)
        k = label.shape[-1] // f8.shape[-1]
        r_cls = F.max_pool2d(label, k)
        r_geo = F.pixel_unshuffle(label.amax(1, keepdim=True), k)
        h = self.rh_body(torch.cat([f8, f16, r_cls, r_geo], 1))
        sup = self.rh_support(h).float()
        claimed = r_cls > 0
        cnt = claimed.flatten(1).sum(1).float()
        has = cnt > 0
        # smooth minimum -logsumexp(-v) over the claimed cells; unclaimed
        # cells enter at a finite -1e4 (not -inf) so a label with nothing
        # drawn backpropagates zeros rather than NaN
        smin = -torch.logsumexp(
            torch.where(claimed, -sup, torch.full_like(sup, -1e4))
            .flatten(1), 1)
        smin = torch.where(has, smin, torch.zeros_like(smin))
        mean = torch.where(claimed, sup, torch.zeros_like(sup)) \
            .flatten(1).sum(1) / cnt.clamp(min=1)
        delta = self.box_mix(torch.stack(
            [smin, mean, torch.log1p(cnt)], 1)).squeeze(1)
        out = logits + torch.stack([torch.zeros_like(delta), delta], 1)
        return (out, sup, self.rh_cls(h)) if aux else out


class CorrNet(nn.Module):
    """Two full ResNet-18 branches with correlation fusion and the per-cell
    label-plane head -- the network this release trains.

    Both branches run to layer4 (stride 32; 7 x 7 for 224-px inputs): the
    image branch is ImageNet-pretrained (fine-tuned by train.py at 0.1 x
    the learning rate, README item 5; its BatchNorm statistics kept at the
    pretrained values, README item 2), the label branch is randomly
    initialised with a num_classes-channel first convolution. The
    classifier reads both branches globally average-pooled and three
    agreement terms between the aligned stride-32 maps (README item 2):
    their element-wise product averaged over the grid, and, on
    L2-normalised features, the product averaged over the cells the label
    draws on (masked) and its per-channel maximum -- 5 x 512 features
    through fully connected layers 512, 256, 128 and 2. With raster_head
    (README item 4), the per-cell head reads the image branch's own layer2
    (stride 8) and layer3 (stride 16) next to the label planes and adds the
    claimed cells' smooth minimum, mean and log(1 + count) to the good
    logit through box_mix, initialised at zero.
    """

    corr = True     # image-grounded fusion (read by interactive_demo.py)

    def __init__(self, num_classes=20, pretrained=True, raster_head=False):
        super().__init__()
        self.raster_head = raster_head
        img = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1
                       if pretrained else None)
        self.model_1 = nn.Sequential(*list(img.children())[:-2])
        lbl = resnet18()
        lbl.conv1 = nn.Conv2d(num_classes, 64, kernel_size=7, stride=2,
                              padding=3, bias=False)
        self.model_2 = nn.Sequential(*list(lbl.children())[:-2])
        if raster_head:
            def block(cin, cout, dil):
                return [nn.Conv2d(cin, cout, 3, padding=dil, dilation=dil),
                        nn.GroupNorm(16, cout), nn.ReLU()]
            self.rh_body = nn.Sequential(
                *block(128 + 256 + num_classes + 64, 256, 1),
                *block(256, 256, 2), *block(256, 256, 4),
                *block(256, 128, 8))
            self.rh_support = nn.Conv2d(128, num_classes, 1)
            self.rh_cls = nn.Conv2d(128, num_classes + 1, 1)
            self.box_mix = nn.Linear(3, 1)
            nn.init.zeros_(self.box_mix.weight)
            nn.init.zeros_(self.box_mix.bias)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 5, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 2))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # the pretrained image branch keeps its BatchNorm statistics
            # (README item 2); its weights are fine-tuned (item 5)
            for m in self.model_1.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    @staticmethod
    def _box_masked(prod, label):
        """Average the agreement over the cells the label draws on."""
        stride = label.shape[-1] // prod.shape[-1]
        mask = (F.max_pool2d(label.amax(1, keepdim=True), stride)
                > 0).to(prod.dtype)
        return (prod * mask).sum(dim=(2, 3)) / \
            mask.sum(dim=(2, 3)).clamp(min=1.0)

    def forward(self, img, label, aux=False):
        """aux=True (a raster_head model) also returns the head's per-cell
        support (B, num_classes, g, g) and class logits
        (B, num_classes + 1, g, g), which its training losses read."""
        # resnet18's children: [:5] ends after layer1 (stride 4), then
        # layer2 (stride 8), layer3 (stride 16), layer4 (stride 32)
        i8 = self.model_1[5](self.model_1[:5](img))
        i16 = self.model_1[6](i8)
        f_img = self.model_1[7](i16)
        f_lbl = self.model_2(label)
        cos = F.normalize(f_img, dim=1) * F.normalize(f_lbl, dim=1)
        parts = [f_img.mean(dim=(2, 3)), f_lbl.mean(dim=(2, 3)),
                 (f_img * f_lbl).mean(dim=(2, 3)),
                 self._box_masked(cos, label), cos.amax(dim=(2, 3))]
        logits = self.fc(torch.cat(parts, 1))
        if not self.raster_head:
            return logits
        return self._raster(logits, label, i8, i16, aux)

    def _raster(self, logits, label, i8, i16, aux=False):
        """The label-plane head: per-cell class support and class, and the
        claimed cells' support mixed into the good logit."""
        k = label.shape[-1] // i8.shape[-1]
        r_cls = F.max_pool2d(label, k)
        r_geo = F.pixel_unshuffle(label.amax(1, keepdim=True), k)
        h = self.rh_body(torch.cat(
            [i8, F.interpolate(i16, size=i8.shape[-2:], mode="bilinear",
                               align_corners=False),
             r_cls.to(i8.dtype), r_geo.to(i8.dtype)], 1))
        sup = self.rh_support(h).float()
        claimed = r_cls > 0
        cnt = claimed.flatten(1).sum(1).float()
        has = cnt > 0
        # smooth minimum -logsumexp(-v) over the claimed cells; unclaimed
        # cells enter at a finite -1e4 (not -inf) so a label with nothing
        # drawn backpropagates zeros rather than NaN
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


class _GhostBNMixin:
    """BatchNorm whose training statistics are computed per virtual
    sub-batch of virtual_bs samples (train.py --ghost_bn). Eval behaviour
    is standard, so the image branch's BatchNorm layers, which CorrNet
    keeps in eval mode, use their pretrained statistics."""

    def forward(self, x):
        vb = self.virtual_bs
        if not self.training or x.shape[0] <= vb or x.shape[0] % vb != 0:
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
    """Replace every BatchNorm1d/2d in module with its ghost variant,
    keeping weights, statistics and flags (state dicts are unchanged)."""
    for name, child in module.named_children():
        cls = None
        if isinstance(child, nn.BatchNorm2d) and \
                not isinstance(child, GhostBatchNorm2d):
            cls = GhostBatchNorm2d
        elif isinstance(child, nn.BatchNorm1d) and \
                not isinstance(child, GhostBatchNorm1d):
            cls = GhostBatchNorm1d
        if cls is not None:
            g = cls(child.num_features, virtual_bs, eps=child.eps,
                    momentum=child.momentum, affine=child.affine,
                    track_running_stats=child.track_running_stats)
            g.load_state_dict(child.state_dict())
            g.training = child.training
            setattr(module, name, g)
        else:
            ghostify(child, virtual_bs)
    return module


def load_model(ckpt_path, device):
    """Construct the network a checkpoint holds (CorrNet, or the listings'
    NetSmall and its correlation variant) and load it."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    rh = any(k.startswith("rh_") for k in sd)
    # CorrNet's fc starts Flatten, Linear (fc.1 is 2-D); NetSmall's with a
    # Linear then BatchNorm1d (fc.1 is 1-D)
    if sd["fc.1.weight"].dim() == 2:
        model = CorrNet(num_classes=sd["model_2.0.weight"].shape[1],
                        pretrained=False, raster_head=rh)
    else:
        corr = sd["fc.0.weight"].shape[1] > 128
        model = NetSmall(corr=corr, pretrained=False, raster_head=rh)
    model.to(device).load_state_dict({k: v.to(device) for k, v in sd.items()})
    model.eval()
    return model
