#!/usr/bin/env python3
"""
phototriage — find the blurry and near-duplicate frames in a shoot, and file the rest by content.

Two phases, deliberately separated:

    scan   measures every image and writes a reviewable plan (HTML + CSV + JSON).
           Touches no originals.
    apply  executes that plan. Moves by default; deleting requires --action delete --yes.

The AI is open-weight and runs on your own machine. Nothing is uploaded: with
--ai the only network access is the one-time model download to ~/.cache/huggingface.

  python phototriage.py scan  ~/Pictures/WeddingSmith --ai
  open ~/Pictures/WeddingSmith/../.phototriage/report.html      # review the plan
  python phototriage.py apply ~/Pictures/WeddingSmith

Categories from an open CLIP checkpoint (openai/clip-vit-base-patch32, MIT-licensed weights)
are metadata only — they are used to *organise and label*, never to decide what gets deleted.
Deletion comes from measured sharpness and near-duplicate grouping, both of which are
explainable in one sentence per file.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import hashlib
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

SCHEMA_VERSION = 3
MANIFEST_NAME = "manifest.json"
QUARANTINE_DIR = "_FirstPass_rejected"
OUT_DIR_NAME = ".phototriage"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
# RAW formats are intentionally out of scope: they need demosaicing to score honestly.

FOCUS_MAX_DIM = 1024          # sharpness is measured on this normalised size
TILE = 32                     # tile edge in pixels for the per-tile focus map
TOP_FRACTION = 0.02           # share of the sharpest tiles that form the focus score
THUMB_MAX_DIM = 360
THUMB_QUALITY = 72

DEFAULT_MODEL = "openai/clip-vit-base-patch32"

# Two frames only count as the same shot if BOTH signals agree: a close perceptual hash
# and a close shape correlation. The hash alone is not enough — dHash averages a frame
# down to 9x8, so unrelated high-frequency frames (foliage, gravel, confetti) collapse to
# near-identical hashes. Without the second signal those unrelated frames get deleted as
# "duplicates" of each other, which is the worst bug this tool could have.
SIGNATURE_SIZE = 16
SIGNATURE_MIN_CORRELATION = 0.92
HASH_MAX_DISTANCE = 10

# Default labels. Override with --categories or --categories-file.
DEFAULT_CATEGORIES = [
    "a portrait of a person or a small group",
    "two people posing together",
    "a large group photo",
    "a wedding ceremony",
    "a party or reception with dancing",
    "candid guests talking and laughing",
    "decorations, table settings and details",
    "food, drink or a cake",
    "a building, venue or landscape",
    "a pet or an animal",
    "a close-up of an object or product",
]

# Sensitivity presets: (relative_fraction_of_batch_median, absolute_floor)
#
# A file is flagged blurry only when its focus score falls below the guard
# max(floor, frac * batch_median) — it has to be bad relative to the rest of the shoot
# *and* bad in absolute terms. Erring towards keeping a soft frame is the whole point: a
# retained blur is an inconvenience, a deleted keeper is a lost wedding.
#
# Calibrated against the synthetic shoot in scripts/make_fixtures.py, where the gap is
# wide: gaussian radius-3 and radius-14 frames score 3.0-13.5, sharp frames 1,541-4,792,
# and deliberate shallow-depth-of-field frames 31,219-42,262. Medium therefore sits at
# 0.08 x median (about 247 on that shoot), roughly 18x above the worst reject and 6x
# below the weakest keeper. Real shoots are narrower than that; use --min-focus or the
# printed distribution when a batch does not separate this cleanly.
SENSITIVITY = {
    "low": (0.04, 15.0),
    "medium": (0.08, 25.0),
    "high": (0.15, 45.0),
}

VERDICT_KEEP = "keep"
VERDICT_BLURRY = "reject:blurry"
VERDICT_DUPLICATE = "reject:duplicate"
VERDICT_UNREADABLE = "skip:unreadable"


# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------


@dataclass
class Record:
    path: str                       # as given, relative to the scanned root
    name: str
    size: int                       # bytes at scan time — used to detect later modification
    mtime: float
    width: int = 0
    height: int = 0
    taken_at: float = 0.0           # EXIF DateTimeOriginal, else mtime
    taken_source: str = ""          # "exif" | "mtime"
    focus: float = 0.0              # mean of the sharpest 2% of tiles — the decision variable
    focus_p90: float = 0.0          # 90th percentile tile, kept as a diagnostic
    focus_median: float = 0.0       # typical tile
    focus_ratio: float = 0.0        # focus/median: high means shallow depth of field
    dhash: str = ""                 # 64-bit perceptual hash, hex
    sig: str = ""                   # 16x16 greyscale fingerprint, hex, for the guard
    verdict: str = VERDICT_KEEP
    reason: str = ""
    group_id: int = -1
    group_size: int = 0
    category: str = ""
    category_score: float = 0.0
    category_runner_up: str = ""
    category_runner_up_score: float = 0.0
    thumb: str = ""
    error: str = ""

    @property
    def is_reject(self) -> bool:
        return self.verdict.startswith("reject:")


@dataclass
class RunMeta:
    schema_version: int = SCHEMA_VERSION
    generated_at: float = field(default_factory=time.time)
    root: str = ""
    sensitivity: str = "medium"
    threshold: float = 0.0
    batch_median_focus: float = 0.0
    ai_enabled: bool = False
    model: str = ""
    device: str = ""
    categories: list[str] = field(default_factory=list)
    dedupe: str = "hash"
    burst_window: float = 3.0
    counts: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


# --------------------------------------------------------------------------------------
# Focus / sharpness
# --------------------------------------------------------------------------------------


def focus_scores(gray: "np.ndarray") -> tuple[float, float, float]:
    """Per-tile variance of a Laplacian, summarised as (focus, p90, median).

    Why tiles rather than one number for the whole frame: a portrait shot at f/1.4 has a
    tack-sharp face and a creamy background. Averaged globally it looks "soft" and a naive
    blur detector deletes the best photo of the day, so the decision uses only the sharpest
    region.

    Why the mean of the sharpest 2% rather than a 90th percentile: a percentile only
    rescues a sharp region that covers more than its own share of the frame. At p90 that
    means a subject has to be sharp across over 10% of the tiles, and a small off-centre
    subject — measured here at about 3% — falls through and the frame is deleted as blurry.
    Averaging the top 2% keeps the rescue working for small subjects while staying stable
    against a single noisy tile, which a plain max() would not.
    """
    import numpy as np

    g = gray.astype(np.float32)
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    )
    lap = lap[1:-1, 1:-1]

    tile = TILE if min(lap.shape) >= TILE * 4 else max(8, min(TILE, min(lap.shape) // 4))
    ny, nx = lap.shape[0] // tile, lap.shape[1] // tile
    if ny < 1 or nx < 1:
        whole = float(lap.var())
        return whole, whole, whole

    cropped = lap[: ny * tile, : nx * tile]
    blocks = cropped.reshape(ny, tile, nx, tile).transpose(0, 2, 1, 3)
    variances = blocks.reshape(ny * nx, -1).var(axis=1)
    top_n = max(1, int(round(variances.size * TOP_FRACTION)))
    focus = float(np.sort(variances)[-top_n:].mean())
    return focus, float(np.percentile(variances, 90)), float(np.median(variances))


def dhash(image, size: int = 8) -> str:
    """64-bit difference hash. Dependency-free near-duplicate signal."""
    import numpy as np

    small = image.convert("L").resize((size + 1, size), 1)  # 1 == PIL LANCZOS constant
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(hex_a: str, hex_b: str) -> int:
    if not hex_a or not hex_b:
        return 64
    return bin(int(hex_a, 16) ^ int(hex_b, 16)).count("1")


def signature(image, size: int = SIGNATURE_SIZE) -> str:
    """Mean-centred, scale-normalised greyscale fingerprint of the frame.

    Stored so duplicate claims can be checked without decoding the file a second time.
    Mean-centring and normalising by spread make it insensitive to exposure and contrast,
    which is exactly right here: two frames of one shot differ in exposure, not in shape.
    """
    import numpy as np
    from PIL import Image

    small = image.convert("L").resize((size, size), Image.Resampling.BOX)
    arr = np.asarray(small, dtype=np.float32).ravel()
    arr = arr - arr.mean()
    span = float(np.abs(arr).max()) or 1.0
    quantised = np.clip((arr / span) * 127.0 + 128.0, 0, 255).astype("uint8")
    return quantised.tobytes().hex()


def correlate(hex_a: str, hex_b: str) -> float:
    """Pearson correlation of two stored signatures, in [-1, 1]. 0.0 when unusable."""
    import numpy as np

    if not hex_a or not hex_b or len(hex_a) != len(hex_b):
        return 0.0
    try:
        a = np.frombuffer(bytes.fromhex(hex_a), dtype="uint8").astype("float32")
        b = np.frombuffer(bytes.fromhex(hex_b), dtype="uint8").astype("float32")
    except ValueError:
        return 0.0
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


# --------------------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------------------


def load_record(path: Path, rel: str, root: Path, thumbs_dir: Path | None) -> Record:
    import numpy as np
    from PIL import Image, ImageOps

    stat = path.stat()
    rec = Record(path=rel, name=path.name, size=stat.st_size, mtime=stat.st_mtime)
    try:
        with Image.open(path) as img:
            del root  # kept in the signature for callers that want to report a relative path
            img = ImageOps.exif_transpose(img)
            rec.width, rec.height = img.size
            rec.taken_at, rec.taken_source = exif_time(img, stat.st_mtime)

            focus_img = img.convert("L")
            # BOX (an area average) rather than LANCZOS: a sharpening resampler rings at
            # edges and can make a moderately blurred frame score as if it were crisp.
            focus_img.thumbnail((FOCUS_MAX_DIM, FOCUS_MAX_DIM), Image.Resampling.BOX)
            rec.focus, rec.focus_p90, rec.focus_median = focus_scores(np.asarray(focus_img))
            rec.focus_ratio = rec.focus / rec.focus_median if rec.focus_median else 0.0
            rec.dhash = dhash(img)
            rec.sig = signature(img)

            if thumbs_dir is not None:
                thumb = img.convert("RGB")
                thumb.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM), Image.Resampling.LANCZOS)
                name = f"{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:16]}.jpg"
                thumb.save(thumbs_dir / name, "JPEG", quality=THUMB_QUALITY, optimize=True)
                rec.thumb = f"thumbs/{name}"
    except (OSError, ValueError) as exc:
        # A corrupt or truncated file must not kill a 3,000-frame run. Only decoding
        # failures are caught here — a NameError in the measurement code is a bug, and
        # letting it masquerade as "unreadable file" would hide it behind a plausible verdict.
        rec.verdict = VERDICT_UNREADABLE
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.reason = "could not be decoded"
    return rec


EXIF_DATETIME_ORIGINAL = 36867
EXIF_DATETIME = 306


def exif_time(img, fallback: float) -> tuple[float, str]:
    """EXIF DateTimeOriginal -> epoch. Falls back to file mtime."""
    try:
        exif = img.getexif()
        raw = None
        ifd = exif.get_ifd(0x8769)  # Exif sub-IFD
        if ifd:
            raw = ifd.get(EXIF_DATETIME_ORIGINAL)
        if not raw:
            raw = exif.get(EXIF_DATETIME)
        if raw:
            stamp = re.sub(r"[^0-9]", "", str(raw))[:14]
            if len(stamp) == 14:
                parsed = time.mktime(time.strptime(stamp, "%Y%m%d%H%M%S"))
                return parsed, "exif"
    except Exception:
        pass
    return fallback, "mtime"


# --------------------------------------------------------------------------------------
# AI: open-weight CLIP for categories + embeddings
# --------------------------------------------------------------------------------------


def projected_embedding(output, projection_dim: int):
    """Coerce whatever get_text_features/get_image_features returned into (N, projection_dim).

    transformers 4.x returned a bare tensor; 5.x returns a ModelOutput whose pooler_output
    holds the projection. Rather than trusting either shape, check the width and fail loudly:
    a silent mismatch here would compare image and text vectors living in different spaces,
    and every category downstream would be quietly wrong.
    """
    import torch

    if torch.is_tensor(output):
        tensor = output
    else:
        tensor = None
        for attr in ("pooler_output", "image_embeds", "text_embeds"):
            candidate = getattr(output, attr, None)
            if torch.is_tensor(candidate):
                tensor = candidate
                break
    if tensor is None:
        raise RuntimeError(f"no embedding tensor found in {type(output).__name__}")
    if tensor.shape[-1] != projection_dim:
        raise RuntimeError(
            f"embedding width {tensor.shape[-1]} != projection_dim {projection_dim}: "
            f"the model returned an unprojected {type(output).__name__} and image/text "
            "features would not be comparable"
        )
    return tensor


class ClipTagger:
    """Zero-shot categories and image embeddings from an open CLIP checkpoint.

    Categories are a *label*, not a decision. Nothing here can cause a file to be deleted.
    """

    def __init__(self, model_id: str, labels: list[str], device: str, batch_size: int, offline: bool):
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.labels = labels
        self.projection_dim = int(self.model.config.projection_dim)
        # CLIP's learned temperature. Read it from the checkpoint rather than assume 100.
        self.logit_scale = 100.0
        if hasattr(self.model, "logit_scale"):
            with torch.no_grad():
                self.logit_scale = float(self.model.logit_scale.exp())

        prompts = [f"a photo of {label}" for label in labels]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = projected_embedding(
                self.model.get_text_features(**inputs), self.projection_dim
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features

    def run(self, paths: list[Path], batch_size: int | None = None):
        """Score image paths in bounded batches.

        Frames are decoded one batch at a time and released immediately. Holding 3,000
        full-resolution images in memory is how this tool would die on a real shoot.
        `draft` lets Pillow decode a JPEG at reduced scale where the format allows it,
        so the second decode is much cheaper than the first.
        """
        import numpy as np
        from PIL import Image, ImageOps

        size = batch_size or self.batch_size
        embeddings, probs = [], []
        for start in range(0, len(paths), size):
            images = []
            for path in paths[start : start + size]:
                with Image.open(path) as img:
                    img.draft("RGB", (512, 512))
                    images.append(ImageOps.exif_transpose(img).convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                features = projected_embedding(
                    self.model.get_image_features(**inputs), self.projection_dim
                )
                features = features / features.norm(dim=-1, keepdim=True)
                logits = self.logit_scale * features @ self.text_features.T
                batch_probs = logits.softmax(dim=-1)
            embeddings.append(features.cpu().float().numpy())
            probs.append(batch_probs.cpu().float().numpy())
            del images, inputs, features, logits, batch_probs
        if not embeddings:
            return np.zeros((0, 0), dtype="float32"), np.zeros((0, len(self.labels)), dtype="float32")
        return np.vstack(embeddings), np.vstack(probs)


# --------------------------------------------------------------------------------------
# Grouping and verdicts
# --------------------------------------------------------------------------------------


def group_bursts(records: list[Record], window: float, mode: str,
                 embeddings=None, hash_distance: int = HASH_MAX_DISTANCE,
                 cosine: float = 0.95, min_correlation: float = SIGNATURE_MIN_CORRELATION) -> None:
    """Group frames that are the same shot taken seconds apart. Mutates records in place."""
    candidates = [r for r in records if r.verdict != VERDICT_UNREADABLE]
    candidates.sort(key=lambda r: r.taken_at)

    group_id = 0
    current: list[Record] = []

    def similar(a: Record, b: Record) -> bool:
        if mode == "embedding" and embeddings is not None:
            import numpy as np

            return float(np.dot(embeddings[a._idx], embeddings[b._idx])) >= cosine
        if hamming(a.dhash, b.dhash) > hash_distance:
            return False
        # the guard: the hash must be corroborated by matching shape
        return correlate(a.sig, b.sig) >= min_correlation

    def flush():
        nonlocal group_id, current
        if not current:
            return
        for rec in current:
            rec.group_id = group_id
            rec.group_size = len(current)
        group_id += 1
        current = []

    for rec in candidates:
        if current and rec.taken_at - current[-1].taken_at <= window and similar(current[-1], rec):
            current.append(rec)
        elif current and rec.taken_at - current[-1].taken_at <= window and any(similar(m, rec) for m in current):
            current.append(rec)
        else:
            flush()
            current = [rec]
    flush()


def decide(records: list[Record], sensitivity: str, threshold_override: float | None,
           mode: str) -> tuple[RunMeta, float]:
    """Assign verdicts. Conservative by construction: every rejection must be justifiable."""
    dedupe = mode != "off"
    live = [r for r in records if r.verdict != VERDICT_UNREADABLE]
    scores = sorted(r.focus for r in live)
    median = scores[len(scores) // 2] if scores else 0.0

    frac, floor = SENSITIVITY[sensitivity]
    threshold = threshold_override if threshold_override is not None else max(floor, frac * median)

    for rec in live:
        if rec.focus < threshold:
            rec.verdict = VERDICT_BLURRY
            rec.reason = (
                f"focus {rec.focus:.1f} below threshold {threshold:.1f} "
                f"(shoot median {median:.1f})"
            )

    if dedupe:
        groups: dict[int, list[Record]] = {}
        for rec in live:
            if rec.group_size > 1:
                groups.setdefault(rec.group_id, []).append(rec)
        for members in groups.values():
            # Best frame of the burst is simply the sharpest; that is a rule a
            # photographer can predict and disagree with, which matters more here
            # than a model score nobody can interrogate.
            best = max(members, key=lambda r: r.focus)
            for rec in members:
                if rec is best or rec.verdict == VERDICT_BLURRY:
                    continue
                rec.verdict = VERDICT_DUPLICATE
                gap = best.focus - rec.focus
                rec.reason = (
                    f"near-duplicate of {best.name} in a burst of {len(members)} "
                    f"({gap:.0f} focus points behind it)"
                )

    meta = RunMeta(
        root="",
        sensitivity=sensitivity,
        threshold=threshold,
        batch_median_focus=median,
        dedupe=mode,
    )
    return meta, median


# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------


def write_manifest(out_dir: Path, meta: RunMeta, records: list[Record]) -> None:
    payload = {
        "meta": asdict(meta),
        "records": [asdict(r) for r in records],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(out_dir: Path, records: list[Record]) -> None:
    columns = [
        "name", "path", "verdict", "reason", "category", "category_score",
        "category_runner_up", "category_runner_up_score", "focus", "focus_p90",
        "focus_median", "group_id", "group_size", "taken_at", "width", "height",
        "size", "thumb",
    ]
    with (out_dir / "triage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def write_report(out_dir: Path, meta: RunMeta, records: list[Record]) -> None:
    sections = [
        ("Would be rejected — blurry", VERDICT_BLURRY),
        ("Would be rejected — near-duplicate", VERDICT_DUPLICATE),
        ("Could not be read", VERDICT_UNREADABLE),
        ("Kept", VERDICT_KEEP),
    ]

    def card(rec: Record) -> str:
        thumb = html.escape(rec.thumb) if rec.thumb else ""
        image = (
            f'<img loading="lazy" src="{thumb}" alt="">' if thumb
            else '<div class="noimg">no thumbnail</div>'
        )
        category = (
            f'<span class="cat">{html.escape(rec.category)} '
            f'<b>{rec.category_score:.0%}</b></span>' if rec.category else ""
        )
        return f"""      <figure class="card">
        {image}
        <figcaption>
          <span class="name">{html.escape(rec.name)}</span>
          {category}
          <span class="why">{html.escape(rec.reason)}</span>
          <span class="num">focus {rec.focus:.1f} · p90 {rec.focus_p90:.1f} · median {rec.focus_median:.1f} · ratio {rec.focus_ratio:.1f}</span>
        </figcaption>
      </figure>"""

    blocks = []
    for title, verdict in sections:
        members = [r for r in records if r.verdict == verdict]
        if not members:
            continue
        if verdict == VERDICT_KEEP and len(members) > 400:
            members = sorted(members, key=lambda r: r.focus)[:400]
            note = "<p class='note'>showing the 400 softest kept frames</p>"
        else:
            note = ""
        cards = "\n".join(card(r) for r in members)
        blocks.append(
            f"  <section>\n    <h2>{html.escape(title)} <span class='count'>{len([r for r in records if r.verdict == verdict])}</span></h2>\n"
            f"    {note}\n    <div class='grid'>\n{cards}\n    </div>\n  </section>"
        )

    counts = meta.counts
    summary = " · ".join(f"{k}: {v}" for k, v in counts.items())
    ai_line = (
        f"categories from <code>{html.escape(meta.model)}</code> on <code>{html.escape(meta.device)}</code>"
        if meta.ai_enabled else "AI categorization was not enabled for this run"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>phototriage — review before you delete</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:2rem; background:#111214; color:#e7e7ea;
         font:15px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }}
  h1 {{ font-size:1.4rem; margin:0 0 .25rem; }}
  h2 {{ font-size:1rem; margin:2.5rem 0 .75rem; padding-bottom:.4rem;
        border-bottom:1px solid #2a2b30; }}
  .count {{ color:#8b8d97; font-weight:400; }}
  .meta {{ color:#8b8d97; font-size:.85rem; margin:0 0 .25rem; }}
  .grid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); }}
  .card {{ margin:0; }}
  .card img, .noimg {{ width:100%; aspect-ratio:3/2; object-fit:cover;
        border-radius:6px; background:#1b1c20; display:flex; align-items:center;
        justify-content:center; color:#55575f; font-size:.8rem; }}
  figcaption {{ display:flex; flex-direction:column; gap:.2rem; padding-top:.5rem; font-size:.8rem; }}
  .name {{ color:#fff; font-weight:600; word-break:break-all; }}
  .cat {{ color:#7fd1a8; }}
  .why {{ color:#e0b872; }}
  .num {{ color:#6c6e78; }}
  .note {{ color:#8b8d97; font-size:.85rem; }}
  code {{ background:#1b1c20; padding:.1rem .3rem; border-radius:3px; }}
</style></head>
<body>
  <h1>phototriage review</h1>
  <p class="meta">{html.escape(meta.root)}</p>
  <p class="meta">sensitivity <code>{html.escape(meta.sensitivity)}</code> ·
     blur threshold <code>{meta.threshold:.1f}</code> ·
     shoot median focus <code>{meta.batch_median_focus:.1f}</code> ·
     burst grouping <code>{html.escape(meta.dedupe)}</code> within {meta.burst_window:.0f}s</p>
  <p class="meta">{summary}</p>
  <p class="meta">{ai_line}. <b>Nothing has been moved or deleted yet</b> — this is the plan.</p>
{chr(10).join(blocks)}
</body></html>
"""
    (out_dir / "report.html").write_text(document, encoding="utf-8")


# --------------------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------------------


def discover(root: Path, recursive: bool, limit: int | None) -> list[Path]:
    skip_dirs = {OUT_DIR_NAME, QUARANTINE_DIR}
    files: list[Path] = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(part in skip_dirs for part in path.relative_to(root).parts):
            continue
        files.append(path)
        if limit and len(files) >= limit:
            break
    return files


def cmd_scan(args) -> int:
    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve() if args.out else root / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    thumbs_dir = None
    if not args.no_thumbs:
        thumbs_dir = out_dir / "thumbs"
        thumbs_dir.mkdir(exist_ok=True)

    started = time.time()
    files = discover(root, not args.no_recursive, args.limit)
    if not files:
        print(f"no images found in {root}")
        return 1

    labels = load_categories(args)
    print(f"scanning {len(files)} images in {root}")
    print(f"  output    {out_dir}")
    print(f"  AI        {args.model if (labels and not args.no_ai) else 'off'}")

    records: list[Record] = []
    results: dict[str, Record] = {}

    def work(path: Path) -> tuple[str, Record]:
        rel = str(path.relative_to(root))
        return rel, load_record(path, rel, root, thumbs_dir)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, (rel, rec) in enumerate(pool.map(work, files), 1):
            results[rel] = rec
            if index % 50 == 0 or index == len(files):
                print(f"  measured {index}/{len(files)}", flush=True)

    records = [results[str(p.relative_to(root))] for p in files]
    live = [r for r in records if r.verdict != VERDICT_UNREADABLE]
    # embeddings are indexed by position in `live`, not in `records`
    for idx, rec in enumerate(live):
        rec._idx = idx  # type: ignore[attr-defined]

    # --- AI pass: categories and embeddings -----------------------------------------
    embeddings = None
    meta_ai = (False, "", "")
    if not args.no_ai and labels:
        try:
            tagger = ClipTagger(args.model, labels, args.device, args.batch, args.offline)
            print(f"  categorising {len(live)} frames with {args.model} on {tagger.device}")
            embeddings, probs = tagger.run([root / rec.path for rec in live])
            for idx, rec in enumerate(live):
                order = probs[idx].argsort()[::-1]
                top, second = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
                rec.category = labels[top]
                rec.category_score = float(probs[idx][top])
                rec.category_runner_up = labels[second]
                rec.category_runner_up_score = float(probs[idx][second])
            meta_ai = (True, args.model, tagger.device)
        except Exception as exc:
            print(f"warning: AI categorization unavailable ({type(exc).__name__}: {exc})", file=sys.stderr)
            print("         continuing with blur + duplicate detection only", file=sys.stderr)

    # --- grouping -------------------------------------------------------------------
    mode = "off"
    if args.no_dedupe:
        for rec in records:
            rec.group_id = -1
            rec.group_size = 0
    else:
        mode = "embedding" if (embeddings is not None and args.dedupe == "embedding") else "hash"
        group_bursts(records, args.burst_window, mode, embeddings=embeddings)

    # --- verdicts -------------------------------------------------------------------
    meta, median = decide(records, args.sensitivity, args.min_focus, mode)
    meta.root = str(root)
    meta.ai_enabled, meta.model, meta.device = meta_ai
    meta.categories = labels if meta_ai[0] else []
    meta.burst_window = args.burst_window
    meta.seconds = time.time() - started

    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.verdict] = counts.get(rec.verdict, 0) + 1
    meta.counts = counts

    write_manifest(out_dir, meta, records)
    write_csv(out_dir, records)
    write_report(out_dir, meta, records)

    # --- console summary ------------------------------------------------------------
    scored = sorted(r.focus for r in live)
    if scored:
        pct = lambda q: scored[min(len(scored) - 1, int(len(scored) * q))]
        print("\nfocus distribution (mean of the sharpest 2% of tile Laplacian variance)")
        print(f"  p10 {pct(.1):8.1f}   p25 {pct(.25):8.1f}   p50 {pct(.5):8.1f}   "
              f"p75 {pct(.75):8.1f}   p90 {pct(.9):8.1f}")
        print(f"  threshold {meta.threshold:.1f}  ({meta.sensitivity} sensitivity)")

    print("\nverdicts")
    for verdict in (VERDICT_BLURRY, VERDICT_DUPLICATE, VERDICT_UNREADABLE, VERDICT_KEEP):
        if counts.get(verdict):
            print(f"  {verdict:22} {counts[verdict]:5}")
    print(f"\n{meta.seconds:.1f}s · {len(files) / max(meta.seconds, 0.001):.1f} images/s")
    print(f"\nreview the plan:  file://{out_dir / 'report.html'}")
    print(f"then apply it:    python {Path(sys.argv[0]).name} apply {root}"
          + ("" if meta_ai[0] else "   # add --ai to scan for categories"))
    return 0


def load_categories(args) -> list[str]:
    if args.no_ai:
        return []
    if args.categories_file:
        raw = Path(args.categories_file).expanduser().read_text(encoding="utf-8")
        labels = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("#")]
        return labels or DEFAULT_CATEGORIES
    if args.categories:
        return [c.strip() for c in args.categories.split(",") if c.strip()]
    return DEFAULT_CATEGORIES


# --------------------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------------------


def unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    for counter in range(1, 10_000):
        candidate = dest.with_name(f"{stem}__{counter}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free filename for {dest}")


def cmd_apply(args) -> int:
    root = Path(args.directory).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else root / OUT_DIR_NAME
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"error: no manifest at {manifest_path} — run scan first", file=sys.stderr)
        return 2

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("schema_version") != SCHEMA_VERSION:
        print("error: manifest was written by a different version — re-run scan", file=sys.stderr)
        return 2
    records = [Record(**r) for r in payload["records"]]
    only = {o.strip() for o in args.only.split(",")} if args.only else None
    rejects = [
        r for r in records
        if r.is_reject and (only is None or r.verdict.split(":", 1)[1] in only)
    ]

    action = args.action
    if action == "delete" and not args.yes:
        print("refusing to delete without --yes. Re-run with:  --action delete --yes", file=sys.stderr)
        print(f"({len(rejects)} files would be removed)", file=sys.stderr)
        return 2

    if not rejects:
        print("nothing to do — the plan contains no rejections")
        return 0

    quarantine = root / QUARANTINE_DIR
    moved = deleted = skipped = 0
    freed = 0

    for rec in rejects:
        source = root / rec.path
        if not source.exists():
            print(f"  skip (missing): {rec.path}")
            skipped += 1
            continue
        # The manifest is a snapshot. If a file changed since the scan, the decision we
        # made about it is about a different file — refuse to act on it.
        stat = source.stat()
        if stat.st_size != rec.size or abs(stat.st_mtime - rec.mtime) > 0.5:
            print(f"  skip (modified since scan): {rec.path}")
            skipped += 1
            continue

        if action == "delete":
            if args.dry_run:
                print(f"  would delete {rec.path}")
            else:
                stat_before = source.stat().st_size
                source.unlink()
                freed += stat_before
                deleted += 1
        else:
            bucket = quarantine / rec.verdict.split(":", 1)[1]
            dest = unique_destination(bucket / source.name)
            if args.dry_run:
                print(f"  would {action} {rec.path} -> {dest.relative_to(root)}")
            else:
                bucket.mkdir(parents=True, exist_ok=True)
                if action == "move":
                    shutil.move(str(source), str(dest))
                else:
                    shutil.copy2(str(source), str(dest))
                moved += 1

    if action == "delete":
        print(f"\n{deleted} deleted, {freed / 1_000_000:.1f} MB freed, {skipped} skipped")
    else:
        print(f"\n{moved} {'moved' if action == 'move' else 'copied'} to {quarantine.relative_to(root.parent)}"
              f", {skipped} skipped")
        if action == "move":
            print("originals are recoverable from the quarantine folder until you empty it")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phototriage",
        description="Find blurry and near-duplicate photos, and file the keepers by content. "
                    "Scan first, review, then apply.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  phototriage scan ~/Pictures/2026-08-14-Wedding --ai
  phototriage scan ~/Pictures/shoot --ai --categories-file labels.txt --sensitivity high
  phototriage apply ~/Pictures/shoot --dry-run
  phototriage apply ~/Pictures/shoot --action delete --yes
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="measure every image and write a reviewable plan")
    scan.add_argument("directory")
    scan.add_argument("--out", help=f"output directory (default: <directory>/{OUT_DIR_NAME})")
    scan.add_argument("--ai", dest="no_ai", action="store_false",
                      help="categorize with open CLIP (default on; downloads ~600 MB once)")
    scan.add_argument("--no-ai", dest="no_ai", action="store_true", help="blur + duplicates only")
    scan.set_defaults(no_ai=False)
    scan.add_argument("--model", default=DEFAULT_MODEL, help=f"any HF CLIP checkpoint (default {DEFAULT_MODEL})")
    scan.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    scan.add_argument("--batch", type=int, default=16, help="images per inference batch (default 16)")
    scan.add_argument("--categories", help="comma-separated labels, e.g. 'ceremony,reception,portraits'")
    scan.add_argument("--categories-file", help="file with one label per line")
    scan.add_argument("--offline", action="store_true", help="forbid any network access (needs a cached model)")
    scan.add_argument("--sensitivity", default="medium", choices=list(SENSITIVITY),
                      help="how hard to push on blur (default medium)")
    scan.add_argument("--min-focus", type=float, help="absolute focus threshold, overriding the preset")
    scan.add_argument("--no-dedupe", action="store_true", help="disable burst/near-duplicate grouping")
    scan.add_argument("--dedupe", default="hash", choices=["hash", "embedding"],
                      help="duplicate signal (embedding needs --ai)")
    scan.add_argument("--burst-window", type=float, default=3.0,
                      help="seconds within which two frames can be the same burst (default 3)")
    scan.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4),
                      help="decode/measure threads (default min(8, cpus))")
    scan.add_argument("--no-thumbs", action="store_true", help="skip thumbnails in the HTML report")
    scan.add_argument("--no-recursive", action="store_true")
    scan.add_argument("--limit", type=int, help="stop after N images (for a quick look)")

    apply_cmd = sub.add_parser("apply", help="execute the plan produced by scan")
    apply_cmd.add_argument("directory")
    apply_cmd.add_argument("--out", help=f"where the manifest lives (default: <directory>/{OUT_DIR_NAME})")
    apply_cmd.add_argument("--action", default="move", choices=["move", "copy", "delete"],
                           help="move to quarantine (default), copy, or delete outright")
    apply_cmd.add_argument("--dry-run", action="store_true", help="print what would happen, change nothing")
    apply_cmd.add_argument("--only", help="limit to verdict kinds, e.g. 'blurry' or 'duplicate'")
    apply_cmd.add_argument("--yes", action="store_true", help="required together with --action delete")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.directory).expanduser().resolve()
    # Cheap guardrail: scanning or emptying a whole home or filesystem root is never intended.
    if root in {Path("/"), Path.home()} or str(root) == "/":
        print(f"error: refusing to operate on {root} — point this at a shoot folder", file=sys.stderr)
        return 2
    return cmd_scan(args) if args.command == "scan" else cmd_apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
