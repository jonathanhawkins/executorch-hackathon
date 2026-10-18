"""BiSeNet face-parsing model, vendored from zllrunning/face-parsing.PyTorch (MIT).

`model.py` and `resnet.py` are that repo's files, with two changes, both noted
here so they stay visible:

  * `from resnet import Resnet18` became a relative import, so the files work
    as a package.
  * `import torchvision` was dropped from `model.py`. It was never used, and
    keeping it would have meant installing torchvision on the board — a couple
    of gigabytes of CUDA wheels for an unused import.

Vendored rather than fetched at runtime: the board exports offline, and a model
definition that can change under us is not something to depend on mid-demo.

Trained on CelebAMask-HQ. The 19 classes are indexed as below; the cutout this
project wants is everything except background and clothing.
"""

from __future__ import annotations

import torch

from .model import BiSeNet

__all__ = ["BiSeNet", "CLASSES", "HEAD_CLASSES", "load_face_parser"]

CLASSES = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
    "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck",
    "neck_l", "cloth", "hair", "hat",
]

# Everything that belongs to the person's head: skin, hair, ears, neck, and the
# things worn on it (glasses, earrings, a hat). Only background and `cloth` are
# excluded — `cloth` is exactly the shirt we are replacing.
HEAD_CLASSES = tuple(
    i for i, name in enumerate(CLASSES) if name not in ("background", "cloth")
)


def load_face_parser(checkpoint, n_classes: int = 19) -> BiSeNet:
    """Build BiSeNet with trained weights, without the ImageNet detour.

    `Resnet18.__init__` fetches torchvision's ImageNet resnet18 to initialise
    the backbone. Every one of those weights is about to be overwritten by the
    checkpoint, so the download is pure latency — and it fails outright on a
    board with no route to download.pytorch.org.
    """
    from . import resnet

    original = resnet.Resnet18.init_weight
    resnet.Resnet18.init_weight = lambda self: None
    try:
        net = BiSeNet(n_classes=n_classes)
    finally:
        resnet.Resnet18.init_weight = original

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return net.eval()


class FaceParserForExport(torch.nn.Module):
    """BiSeNet with only its main head, for export.

    The model returns three tensors: the main output and two auxiliary heads
    used to supervise training. Keeping them would triple the output bandwidth
    and give the quantizer two graphs' worth of ranges to fit for no gain.
    """

    def __init__(self, net: BiSeNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0]
