#!/usr/bin/env python3
"""Pre-tokenize the garment catalogue for the billboard's generate mode.

Writes one directory per garment, each holding the tokens_cond.raw and
tokens_uncond.raw that board_sd_driver.py takes, plus a prompts.json index the
daemon and the web page read.

Doing this on the Mac rather than on the board is deliberate: CLIP's vocabulary
and merges stay off the device, and a shop is choosing from a catalogue rather
than typing free text, so there is nothing to tokenize at shutter time.

The catalogue is ControlNet's, so SD 1.5 padding (EOS), not 2.1's.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Deliberately plain descriptions. ControlNet is already holding the pose and
# the garment's silhouette from the photo, so the prompt only has to say what
# the garment IS — and the colour word is the part that actually lands.
CATALOGUE = [
    ("navy-tee", "Navy t-shirt", "a navy blue crew neck t-shirt"),
    ("white-tee", "White t-shirt", "a plain white crew neck t-shirt"),
    ("red-tee", "Red t-shirt", "a bright red crew neck t-shirt"),
    ("black-hoodie", "Black hoodie", "a black pullover hoodie"),
    ("grey-sweatshirt", "Grey sweatshirt", "a heather grey crew neck sweatshirt"),
    ("denim-shirt", "Denim shirt", "a blue denim button-up shirt"),
]

TEMPLATE = "studio photo of a person wearing {}, plain background, soft lighting"
NEGATIVE = (
    "blurry, low quality, distorted, deformed hands, extra limbs, "
    "watermark, text, cropped, out of focus"
)


def main(args):
    here = Path(__file__).resolve().parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = {}
    for slug, label, garment in CATALOGUE:
        prompt = TEMPLATE.format(garment)
        directory = out / slug
        directory.mkdir(exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(here / "make_sd_tokens.py"),
                prompt,
                "--negative-prompt", NEGATIVE,
                "--base", "sd15",
                "-o", str(directory),
            ],
            check=True,
            capture_output=True,
        )
        index[slug] = {"label": label, "garment": garment, "prompt": prompt}
        print(f"  {slug:18s} {label}")

    (out / "prompts.json").write_text(json.dumps(index, indent=2))
    print(f"{len(index)} garments -> {out}/prompts.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default=".tryon_prompts")
    main(parser.parse_args())
