#!/usr/bin/env python3
"""Generate a synthetic demo shoot so phototriage can be exercised without any real photos.

    python scripts/make_fixtures.py /tmp/demo-shoot

Produces a mix that mirrors a real card: bursts of near-identical frames, hard and mild
blur, deliberate shallow-depth-of-field portraits, sharp keepers, and one truncated file
to prove a corrupt image cannot kill a run. Nothing here is derived from a photograph.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

SIZE = (1400, 933)


def scene(seed: int) -> Image.Image:
    """A distinct scene per seed.

    The background has to change with the seed: a fixed shared gradient would make every
    fixture frame a near-duplicate of every other one, and the demo would then exercise a
    detector against images no camera produces.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(25, 220, (5, 7, 3), dtype="uint8")
    img = Image.fromarray(coarse, "RGB").resize(SIZE, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    for _ in range(18):
        x, y = int(rng.integers(0, SIZE[0])), int(rng.integers(0, SIZE[1]))
        w, h = int(rng.integers(80, 320)), int(rng.integers(80, 320))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=3)
        draw.ellipse([x, y, x + w // 2, y + h // 2], fill=colour)
    return img


def texture(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype="uint8")
    return Image.fromarray(arr, "RGB")


def bokeh(seed: int) -> Image.Image:
    """Sharp subject, defocused background — the frame a naive blur detector deletes.

    Both the subject period and the background vary by seed, so these are three different
    photographs that happen to share a shooting style, not three copies of one image.
    """
    rng = np.random.default_rng(seed)
    bg = Image.fromarray(rng.integers(20, 80, (SIZE[1], SIZE[0], 3), dtype="uint8"), "RGB")
    bg = bg.filter(ImageFilter.GaussianBlur(28))
    w = SIZE[0] // (3 + seed % 3)
    h = SIZE[1] // (3 + seed % 3)
    subject = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(subject)
    step = 7 + (seed % 4)
    ink = (200 + seed * 7 % 50, 225, 215)
    for y in range(0, h, step):
        for x in range(0, w, step):
            if (x // step + y // step + seed) % 2 == 0:
                draw.rectangle([x, y, x + step - 1, y + step - 1], fill=ink)
    # off-centre, and a different size per seed: three distinct photographs that happen
    # to share a shooting style, rather than one image pasted three times
    dx = (seed % 5 - 2) * 90
    dy = (seed % 3 - 1) * 70
    bg.paste(subject, ((SIZE[0] - w) // 2 + dx, (SIZE[1] - h) // 2 + dy))
    return bg


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo-shoot").resolve()
    root.mkdir(parents=True, exist_ok=True)
    random.seed(4)

    def save(img: Image.Image, name: str, quality: int = 92) -> None:
        img.save(root / name, "JPEG", quality=quality)

    # 4 bursts of 3 near-identical frames — one of each should survive
    for burst in range(4):
        for frame in range(3):
            save(scene(seed=burst * 10), f"burst{burst}_frame{frame}.jpg")

    # hard blur: the obvious rejects
    for index in range(3):
        save(scene(seed=100 + index).filter(ImageFilter.GaussianBlur(14)), f"blurry_{index}.jpg")

    # mild blur: deliberately ambiguous, this is where sensitivity tuning lives
    for index in range(2):
        save(scene(seed=200 + index).filter(ImageFilter.GaussianBlur(3)), f"soft_{index}.jpg")

    # shallow depth of field: must NOT be rejected
    for index in range(3):
        save(bokeh(seed=300 + index), f"bokeh_{index}.jpg")

    # sharp keepers
    for index in range(4):
        save(texture(seed=400 + index), f"texture_{index}.jpg")
    for index in range(3):
        save(scene(seed=500 + index), f"scene_{index}.jpg")

    # a truncated file: the corrupt-image path
    good = root / "texture_0.jpg"
    (root / "corrupt.jpg").write_bytes(good.read_bytes()[: good.stat().st_size // 3])

    total = len(list(root.glob("*.jpg")))
    print(f"wrote {total} files to {root}")
    print("bursts of 3 · 3 hard blur · 2 mild blur · 3 bokeh · 7 sharp · 1 corrupt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
