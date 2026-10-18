#!/usr/bin/env python3
"""The person-segmentation models, and what their classes mean to this demo.

    python scripts/person_seg.py --list                 # the candidates
    python scripts/person_seg.py frame.jpg -o out/      # fp32 reference, on CPU

Two of the three entries below were exported to the board and timed before one
was picked; the third is listed because it is the obvious next thing to try.
`MODELS` keeps all three so the comparison in docs/personseg-npu.md can be
re-run rather than taken on trust. What the billboard needs from any of them is
two masks:

* a **silhouette** — the whole person, so the garment can be clipped to it and
  stops floating over the background when the placement is a little off;
* **arms** — so a hand or forearm crossing the chest is drawn back over the
  garment instead of disappearing underneath it.

Only the human-parsing model gives the second one. VOC's `person` is a single
blob: an arm in front of a torso is the same class as the torso, so no amount
of post-processing separates them. That, not speed, is what decided this.

Everything here is a plain `torch.nn.Module` so the QNN ahead-of-time compiler
can take it. That ruled out PPHumanSeg, which ships as ONNX only — converting
it back to PyTorch to then re-lower it is a detour with its own failure modes,
for a model that would still only give a silhouette.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

# --- What the classes mean here ---------------------------------------------

# ATR, as used by the segformer_*_clothes checkpoints. Single person, labelled
# into body parts and garments — which is why it can tell an arm from a chest.
ATR_CLASSES = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt",
    "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg",
    "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf",
]

# Pascal VOC 2012, as the bundle's deeplab_v3.py example uses it.
VOC_CLASSES = [
    "Background", "Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus",
    "Car", "Cat", "Chair", "Cow", "DiningTable", "Dog", "Horse", "MotorBike",
    "Person", "PottedPlant", "Sheep", "Sofa", "Train", "TvMonitor",
]


def _indices(classes: list[str], *names: str) -> tuple[int, ...]:
    return tuple(classes.index(n) for n in names)


@dataclass(frozen=True)
class SegModel:
    """A candidate, plus the class bookkeeping the compositor needs."""

    key: str
    classes: list[str]
    size: tuple[int, int]            # (height, width) compiled into the graph
    stride: int                      # logits are input / stride
    params: float                    # millions
    note: str
    silhouette: tuple[int, ...]      # the whole person
    arms: tuple[int, ...] = ()       # occluders in front of the torso
    worn: tuple[int, ...] = ()       # the garment being replaced
    head: tuple[int, ...] = ()

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def logit_shape(self) -> tuple[int, int, int]:
        return (self.n_classes, self.size[0] // self.stride, self.size[1] // self.stride)

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.size[0], self.size[1])

    def load(self) -> torch.nn.Module:
        return _LOADERS[self.key](self)


# `Bag` and `Scarf` are in the silhouette on purpose — a tote held against the
# body is part of the outline, and a garment clipped around it looks wrong.
_ATR_SILHOUETTE = tuple(range(1, len(ATR_CLASSES)))

MODELS: dict[str, SegModel] = {
    "segformer-b0-atr": SegModel(
        key="segformer-b0-atr",
        classes=ATR_CLASSES,
        size=(512, 512),
        stride=4,
        params=3.7,
        note="human parsing; the only candidate that separates arms from torso",
        silhouette=_ATR_SILHOUETTE,
        arms=_indices(ATR_CLASSES, "Left-arm", "Right-arm"),
        worn=_indices(ATR_CLASSES, "Upper-clothes", "Dress"),
        head=_indices(ATR_CLASSES, "Hat", "Hair", "Sunglasses", "Face"),
    ),
    "segformer-b2-atr": SegModel(
        key="segformer-b2-atr",
        classes=ATR_CLASSES,
        size=(512, 512),
        stride=4,
        params=27.4,
        note="same labels as b0, 7x the parameters — the accuracy ceiling",
        silhouette=_ATR_SILHOUETTE,
        arms=_indices(ATR_CLASSES, "Left-arm", "Right-arm"),
        worn=_indices(ATR_CLASSES, "Upper-clothes", "Dress"),
        head=_indices(ATR_CLASSES, "Hat", "Hair", "Sunglasses", "Face"),
    ),
    "dlv3-r101-voc": SegModel(
        key="dlv3-r101-voc",
        classes=VOC_CLASSES,
        size=(224, 224),
        stride=1,
        params=61.0,
        note="the bundle's own deeplab_v3.py example; one person blob, no parts",
        silhouette=_indices(VOC_CLASSES, "Person"),
    ),
}

DEFAULT_MODEL = "segformer-b0-atr"

# --- Loaders -----------------------------------------------------------------

SEGFORMER_REPOS = {
    "segformer-b0-atr": "mattmdjaga/segformer_b0_clothes",
    "segformer-b2-atr": "mattmdjaga/segformer_b2_clothes",
}


def snapshot_path(key: str) -> str:
    """Fetch a SegFormer checkpoint on the Mac, for pushing to the board.

    Weights only — `allow_patterns` keeps the b2 repo's optimiser state and
    ONNX copy out of it, which is most of its size.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        SEGFORMER_REPOS[key],
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
    )


def _load_segformer(model: SegModel) -> torch.nn.Module:
    from transformers import SegformerForSemanticSegmentation

    source = _CHECKPOINT_OVERRIDE or SEGFORMER_REPOS[model.key]
    net = SegformerForSemanticSegmentation.from_pretrained(source)
    return SegformerForExport(net).eval()


def _load_deeplab(model: SegModel) -> torch.nn.Module:
    # torchvision is not a dependency of this repo — it is installed on the
    # board only for this comparison. See board_personseg_export.sh.
    from torchvision.models.segmentation import deeplabv3, deeplabv3_resnet101

    net = deeplabv3_resnet101(weights=deeplabv3.DeepLabV3_ResNet101_Weights.DEFAULT)
    return DeepLabForExport(net).eval()


_LOADERS = {
    "segformer-b0-atr": _load_segformer,
    "segformer-b2-atr": _load_segformer,
    "dlv3-r101-voc": _load_deeplab,
}

_CHECKPOINT_OVERRIDE: str | None = None


def use_local_checkpoint(path: str | Path | None) -> None:
    """Point the SegFormer loaders at a pushed directory instead of the Hub."""
    global _CHECKPOINT_OVERRIDE
    _CHECKPOINT_OVERRIDE = str(path) if path else None


# --- Export wrappers ---------------------------------------------------------


class SegformerForExport(torch.nn.Module):
    """SegFormer returning a bare logits tensor, at its own 1/4 resolution.

    Two things are deliberately left out. `SegformerForSemanticSegmentation`
    returns a `ModelOutput` dataclass, which `torch.export` will not trace; and
    the usual recipe upsamples the logits to the input size before the argmax,
    which is a bilinear resize of 18 channels that changes no argmax outcome a
    host-side resize of the finished mask would not. Keeping the native 128x128
    output cuts what crosses back from the NPU by 16x.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.net(pixel_values=pixel_values).logits


class DeepLabForExport(torch.nn.Module):
    """torchvision's DeepLabV3 without its auxiliary head.

    The model returns `{"out": ..., "aux": ...}`; `aux` supervises training and
    is dead weight here — carrying it would double the output bandwidth and
    hand the quantizer a second graph's worth of ranges to fit.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["out"]


# --- fp32 reference ----------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Person segmentation, fp32 on CPU")
    ap.add_argument("image", nargs="?", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("person_reference"))
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    ap.add_argument("--cover", type=float, default=None,
                    help="body crop size, in face widths")
    ap.add_argument("--full-frame", action="store_true")
    ap.add_argument("--list", action="store_true", help="show the candidates")
    args = ap.parse_args()

    if args.list:
        for model in MODELS.values():
            shape = "x".join(str(v) for v in model.logit_shape)
            print(f"  {model.key:18s} {model.params:5.1f}M  "
                  f"{model.size[1]}x{model.size[0]} -> {shape}")
            print(f"  {'':18s} {model.note}")
        return 0

    if not args.image or not args.image.exists():
        return ap.error("give an image, or --list")

    import time

    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_person_raw import prepare  # noqa: E402

    model = MODELS[args.model]
    net = model.load()
    tensor, crop = prepare(Image.open(args.image), model,
                           cover=args.cover, full_frame=args.full_frame)

    with torch.no_grad():
        inputs = torch.from_numpy(tensor)
        net(inputs)                                   # warm
        start = time.time()
        logits = net(inputs)[0]
        elapsed = time.time() - start

    args.output.mkdir(parents=True, exist_ok=True)
    logits.numpy().astype(np.float32).tofile(args.output / "output_0.raw")
    (args.output / "person_crop.json").write_text(crop.to_json(model))

    print(f"  model  {model.key}  ({model.params:.1f}M params)")
    print(f"  in     {args.image.name}")
    print(f"  infer  {elapsed:.3f} s (fp32, eager, CPU)")
    print(f"  out    {args.output}/output_0.raw  {tuple(logits.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
