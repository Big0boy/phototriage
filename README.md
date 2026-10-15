# phototriage

Find the blurry and near-duplicate frames in a shoot, and file the rest by content — with an
open-weight model running on your own machine.

Nothing is uploaded. The only network access in an `--ai` run is the one-time download of
open CLIP weights to `~/.cache/huggingface`; after that the tool works offline. No photo,
and no thumbnail, and no score ever leaves the computer.

```bash
python scripts/make_fixtures.py /tmp/demo-shoot   # a synthetic shoot: no real photos needed
python phototriage.py scan /tmp/demo-shoot --ai   # measure, categorise, write a plan
# review the plan, then apply it
python phototriage.py apply /tmp/demo-shoot       # moves rejects to a quarantine folder
```

---

## Why two phases

`scan` measures and decides but **changes nothing**. It writes a plan you can read:

| artefact | what it is |
|---|---|
| `report.html` | every frame with its verdict, its reason, and its category — open this first |
| `triage.csv` | the same as a spreadsheet |
| `manifest.json` | the full audit trail, including the thresholds used |
| `thumbs/` | small previews so the report loads instantly |

`apply` executes that plan. The default action **moves** rejects into
`_FirstPass_rejected/<reason>/`, so a mistake is a `mv` back, not a lost wedding. Deleting
requires saying so twice: `--action delete --yes`.

Two more refusals worth knowing about:

- If a file changed after the scan, it is skipped. A plan is a statement about specific
  files; once one has changed, that statement no longer holds.
- `apply` will not run without a manifest, and the manifest is version-checked. It looks in
  `<DIR>/.phototriage/`, so a plan written to a custom `--out` needs the same `--out` here.

---

## What each signal means

**Focus** (`focus`) is the mean variance of a Laplacian over the *sharpest 2% of tiles*, on a
1024px reduction. Tiles, not the whole frame, because a portrait at f/1.4 has a tack-sharp
face and a creamy background: averaged globally it reads as "soft", and a naive detector
deletes the best photo of the day. Averaging the top 2% rather than taking a 90th percentile
matters for small subjects — a percentile only rescues a sharp region covering more than its
own share of the frame, so a subject covering ~3% of it falls through and gets deleted as
blurry. That failure was measured here, which is why the metric is what it is.

`focus_p90`, `focus_median` and `focus_ratio` are reported alongside but do not decide anything.

**Blur threshold** = `max(floor, fraction × median of this shoot)`. A frame has to be bad
relative to its own shoot *and* bad in absolute terms. Presets:

| `--sensitivity` | fraction | floor |
|---|---|---|
| `low` | 0.04 | 15 |
| `medium` (default) | 0.08 | 25 |
| `high` | 0.15 | 45 |

Calibrated on the synthetic shoot in `scripts/make_fixtures.py`, where radius-3 and radius-14
blurs score 3.0–13.5, sharp frames 1,541–4,792, and deliberate shallow-depth-of-field frames
31,219–42,262. Real shoots separate less cleanly than that, so `scan` prints the distribution
and every frame's score. If a batch does not separate, set `--min-focus` directly.

**Duplicates** are grouped by capture time (EXIF, falling back to mtime) plus similarity, and
the sharpest frame in each burst is kept. Similarity needs **both** a close 64-bit dHash and a
correlation of the 16×16 signature — the hash alone is not sufficient. At 9×8, fine texture is
sampled below its own frequency, so two frames of it at different phases hash 8 bits apart
(inside the grouping distance) while correlating at 0.38; without the second signal those are
"duplicates" and a real photo is deleted. Measured, again: same shot → hash 0, correlation
1.00; unrelated sharp frames → hash 13, correlation 0.30.

**Category** comes from open CLIP weights (`openai/clip-vit-base-patch32`, MIT-licensed)
zero-shot against text labels. Categories are a **label, not a decision**: nothing in the
categorical path can cause a file to be deleted. Point it at your own vocabulary:

```bash
python phototriage.py scan ~/Pictures/shoot --ai --categories "ceremony,reception,portraits,details"
python phototriage.py scan ~/Pictures/shoot --ai --categories-file my_labels.txt   # one per line
```

---

## Install

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# or: python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Pillow and numpy are the only hard dependencies; without torch/transformers the tool still
runs with `--no-ai`.

## Use

```bash
phototriage scan  DIR [--ai|--no-ai] [--sensitivity low|medium|high] [--min-focus N]
                       [--categories ...] [--categories-file F] [--model HF_ID]
                       [--device auto|cpu|cuda] [--batch 16] [--offline]
                       [--dedupe hash|embedding] [--no-dedupe] [--burst-window 3]
                       [--jobs N] [--no-thumbs] [--no-recursive] [--limit N] [--out DIR]

phototriage apply DIR [--action move|copy|delete] [--only blurry,duplicate]
                       [--dry-run] [--yes] [--out DIR]
```

`--out` must be passed to **both** commands. If you redirect the plan with
`scan --out elsewhere`, `apply` still looks in `<DIR>/.phototriage/` unless you give it the
same `--out`, and it will refuse to run.

Everyday runs:

```bash
# the honest first run: look at the plan, change nothing
phototriage scan ~/Pictures/2026-08-14-Wedding --ai
phototriage apply ~/Pictures/2026-08-14-Wedding --dry-run

# keep the plan out of the shoot folder — then remember --out on apply too
phototriage scan ~/Pictures/shoot --ai --out ~/Desktop/triage
phototriage apply ~/Pictures/shoot --out ~/Desktop/triage --dry-run

# a shoot that came out soft overall — push harder
phototriage scan ~/Pictures/shoot --sensitivity high

# stop grouping bursts, keep only the blur judgement
phototriage scan ~/Pictures/shoot --no-dedupe

# second opinions: bursts matched by CLIP embeddings rather than the hash
phototriage scan ~/Pictures/shoot --ai --dedupe embedding

# no network at all, model already cached
phototriage scan ~/Pictures/shoot --ai --offline
```

Swapping models is one flag: any Hugging Face CLIP checkpoint works, and the embedding width
is checked at load, so a checkpoint that is not a CLIP projection fails loudly instead of
quietly producing meaningless labels.

## Measured performance

On one laptop CPU (no GPU), against synthetic 1400×933 JPEGs:

| stage | measured |
|---|---|
| CLIP inference | 0.052 s/image (batch 8) |
| blur + hash + signature, no AI | ~43 images/s overall |
| full run, 27 frames incl. model load | 9.7 s |

So ~3,000 frames is in the region of minutes on CPU, and it is inference-bound rather than
decode-bound once `--ai` is on. Real 24MP files decode slower than these fixtures; measure
your own shoot before promising a number.

## Limits

- **RAW is out of scope.** `.cr2/.nef/.arw` need demosaicing to score honestly.
- **No blink detection.** MediaPipe packaging is version-fragile, so it would make the tool
  harder to install than it is useful. "Possible blink" is the obvious next signal.
- **Bursts need time.** Frames without EXIF fall back to file mtime, and a batch copied all at
  once shares an mtime, so unrelated frames can look simultaneous. The similarity test still
  has to pass, but that is why `--no-dedupe` exists.
- **Grouping is conservative, not omniscient.** Frames of one subject differ only in their
  defocused background can still group; the 16×16 signature cannot see past a dominant shared
  structure.
- **Categories are labels, not truth.** CLIP on abstract or unusual content will be wrong.
  Nothing about a category affects what is deleted.

## Tests

```bash
.venv/bin/python -m pytest tests -q     # 37 tests, all fixtures generated in-process
```

The tests generate their own images — this repository contains no photographs — and assert the
contract rather than the tuning: that sharp orders above blurred, that shallow depth of field
is spared, that a hash collision alone never groups two files, that only rejects are touched,
that `copy` leaves the original in place while `move` does not, that `delete` refuses without
`--yes`, that a file modified after the scan is skipped, and that a plan written with
`scan --out` can be read back by `apply --out` (and is refused without it).
