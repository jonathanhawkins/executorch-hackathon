"""GHOST face swap, vendored from ai-forever/ghost (Apache-2.0).

`aei_net.py` and `aad_layer.py` are that repo's `network/AEI_Net.py` and
`network/AADLayer.py`; `iresnet.py` is the ArcFace architecture shipped
alongside the weights in its `arcface` release. Changes, all kept visible:

  * `from .AADLayer import *` became an explicit import of `AAD_ResBlk`, and
    the files were renamed to this repo's snake_case.
  * The `resnet` attribute-encoder backbone was dropped. GHOST's published
    weights are all `unet`, the option needs a third upstream file, and an
    unreachable branch is one more thing for the exporter to trip over.
  * `IResNet.forward` lost its `torch.cuda.amp.autocast(self.fp16)` wrapper.
    There is no CUDA on the board and GHOST builds `iresnet100(fp16=False)`,
    so it was a no-op that `torch.export` would still have traced.

Vendored rather than fetched at runtime, for the same reason as
`vendor/face_parsing`: the board exports offline.

## The two models

`AEI_Net` is the swapper: 184.3 M parameters, taking the *target* face at
256x256 in [-1, 1] plus a 512-d identity vector, returning a 256x256 face in
[-1, 1] through a tanh. `iresnet100` is the identity encoder: 65.2 M
parameters, 112x112 in [-1, 1], 512-d out.

The identity vector is the **raw** ArcFace output, not an L2-normalised one —
`utils/inference/faceshifter_run.py` upstream feeds `netArc`'s output straight
to `G`. Normalising it changes the scale the AAD layers were trained against.

## Licensing, because it is the reason this model was chosen

The repository is Apache-2.0 and the weights are release assets of that same
repository, with no separate licence notice attached to them. That matters:
`inswapper_128`, the better-quality alternative in
[docs/faceswap.md](../../docs/faceswap.md), has explicitly non-commercial
weights, which is a problem for a retail deployment.

The thing worth checking was the identity encoder, since `iresnet.py` is
InsightFace's architecture and InsightFace's *pretrained weights* are
"available for non-commercial research purposes only". Checked:

  * GHOST's `backbone.pth` has the same 925 keys and identical shapes as
    InsightFace's `glint360k_cosface_r100_fp16_0.1`, and is byte-for-byte the
    same length (261,223,796 bytes) — the same iresnet100 architecture.
  * The weights themselves are **uncorrelated**: mean cosine similarity
    0.0008 over 104 weight tensors, max 0.065, none above 0.5. It is not that
    checkpoint and not a fine-tune of it; it is an independent training run.

So the architecture is InsightFace's (whose *code* is MIT), and the weights
are the GHOST authors' own, published under their Apache-2.0 repository. What
this does not prove is that it is not some other r100 from InsightFace's zoo
that was never compared — only that it is not the one everybody redistributes.

SHA256 of the two checkpoints this package expects:

    G_unet_2blocks.pth  85f130421c65722927c47502ea1b28b62be4893ca9180fec735b67be1e5edd8d
    backbone.pth        a566a62357f0c55b679d9ff2f022a294486568be0c00665d39029d0e46a8109b
"""

from __future__ import annotations

import torch

from .aei_net import AEI_Net
from .iresnet import iresnet100

__all__ = ["AEI_Net", "iresnet100", "load_generator", "load_identity_encoder",
           "GeneratorForExport"]


def load_generator(checkpoint, backbone: str = "unet", num_blocks: int = 2) -> AEI_Net:
    """AEI-Net with trained weights. `c_id=512` — upstream's inference default.

    The constructor's own default is 256, which loads nothing and fails
    loudly; GHOST's `inference.py` passes 512 explicitly.
    """
    net = AEI_Net(backbone, num_blocks=num_blocks, c_id=512)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return net.eval()


def load_identity_encoder(checkpoint) -> torch.nn.Module:
    """ArcFace iresnet100, 112x112 in, 512-d out."""
    net = iresnet100(fp16=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    return net.eval()


class GeneratorForExport(torch.nn.Module):
    """AEI-Net returning only the swapped face.

    `AEI_Net.forward` returns `(Y, attr)` — the image plus the eight attribute
    tensors the generator consumed. Those are an internal artifact of the
    U-Net, useful for the training losses and nothing else at inference. They
    are also enormous: 1024x2x2 up to 64x256x256, about 21 MB a frame in
    float32, which on this board is a copy off the NPU that dwarfs the 768 kB
    of actual output. Same reason `FaceParserForExport` drops BiSeNet's
    auxiliary heads.
    """

    def __init__(self, net: AEI_Net):
        super().__init__()
        self.net = net

    def forward(self, target: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.net(target, identity)[0]
