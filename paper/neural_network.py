"""The paper's two-branch label-quality network.

Pičuljan, N. and Car, Ž.: "Machine Learning-Based Label Quality Assurance
for Object Detection Projects in Requirements Engineering", Applied Sciences
13(10):6234, 2023. https://doi.org/10.3390/app13106234

The image and the rendered label are encoded by two ResNet18 trunks,
global-average-pooled, concatenated, and classified into (bad, good) logits.
Apply softmax for confidences; the paper optimises the cross-entropy loss,
read here as two-class cross-entropy on these logits (README
interpretation 1). The released model is CorrNet (coco_corrnet.py), which
builds its label trunk and the image trunk's layout from this module and
widens the fully connected entry layer for its correlation features.
"""

import torch
from torch import nn
from torchvision.models import resnet18

NUM_CLASSES = 80


def _trunk(in_channels):
    m = resnet18(weights=None)
    if in_channels != 3:
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
    return nn.Sequential(*list(m.children())[:-2])


class Net(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_1 = _trunk(3)             # image branch
        self.model_2 = _trunk(NUM_CLASSES)   # label-raster branch
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(),
            nn.Linear(128, 2))
        if device is not None:
            self.to(device)

    def forward(self, cats, background):
        f_img = nn.functional.adaptive_avg_pool2d(self.model_1(background), 1)
        f_lbl = nn.functional.adaptive_avg_pool2d(self.model_2(cats), 1)
        return self.fc(torch.cat([f_img, f_lbl], dim=1))
