# phototriage (FirstPass)

> **3,000 frames come off the card. The photographer only wants to look at 400.**  
> Find blurry and near-duplicate photos, and categorize keepers with an open-weight model running entirely on your own machine.

Nothing is uploaded. The only optional network access in an `--ai` run is the one-time download of open CLIP weights to `~/.cache/huggingface/`. After that, the tool works 100% offline. No client photo, thumbnail, or score ever leaves your drive.

---

## Who is this for?

- **Solo wedding & event photographers** who shoot 2,000–4,000 frames per event and spend 4–8 unpaid hours per shoot manually culling near-identical burst frames.
- **Privacy-conscious professionals:** Client contracts increasingly prohibit uploading unreleased photos to third-party cloud AI vendors. **phototriage** runs entirely locally — satisfying privacy clauses with zero marginal cost per image.
- **Anyone with large photo shoots** looking for an automated, explainable first pass that separates technical failures from keepers without making subjective taste decisions.

---

## Quick Start (Test in 2 Minutes)

You don't need real photos to test phototriage. The repository includes a deterministic synthetic shoot generator:

```bash
# 1. Clone & install
git checkout main
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Generate a synthetic test shoot (28 images: blurs, bursts, bokeh, sharp keepers)
python scripts/make_fixtures.py /tmp/demo-shoot

# 3. Scan the shoot and generate the review plan
phototriage scan /tmp/demo-shoot --ai

# 4. Open the visual HTML plan in your browser
open /tmp/demo-shoot/.phototriage/report.html       # macOS
xdg-open /tmp/demo-shoot/.phototriage/report.html   # Linux

# 5. Preview the actions (dry run — changes nothing)
phototriage apply /tmp/demo-shoot --dry-run

# 6. Apply the plan (moves rejects safely to quarantine)
phototriage apply /tmp/demo-shoot --action move
```

---

## Installation

### Prerequisites
- Python 3.9+
- Linux or macOS (Windows supported via WSL or PowerShell)

### Using `uv` (Fastest)
```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

### Standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

*Note: For GPU acceleration on Linux/Windows, install PyTorch with CUDA support prior to running:*
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## How to Use phototriage

phototriage uses a strict **two-phase workflow** so nothing is ever deleted or moved without your review:

```
[ Camera JPEGs ] ──> phototriage scan ──> Review report.html ──> phototriage apply ──> [ Keepers in place ]
                                                                                   └──> [ Rejects in quarantine ]
```

### Step 1: Scan (`phototriage scan`)

Measures every photo, clusters bursts, categorises with open CLIP, and writes a reviewable plan. **Original files are untouched.**

```bash
# Standard scan with local AI categorisation
phototriage scan ~/Pictures/2026-Wedding-Smith --ai

# Fast scan without AI (blur & duplicate detection only)
phototriage scan ~/Pictures/2026-Wedding-Smith --no-ai

# Scan with custom event labels
phototriage scan ~/Pictures/2026-Wedding-Smith --ai --categories "ceremony,cocktails,dance,portraits"

# Increase blur sensitivity (low, medium [default], high)
phototriage scan ~/Pictures/2026-Wedding-Smith --sensitivity high
```

#### What `scan` generates in `<directory>/.phototriage/`:
- `report.html`: Visual report showing every photo, its verdict, its sharpness metrics, and category badge. Open this first!
- `triage.csv`: A spreadsheet with all scores and verdicts for bulk analysis.
- `manifest.json`: Full machine-readable audit trail and decision parameters.
- `thumbs/`: Low-resolution previews so `report.html` opens instantly.

### Step 2: Review

Open `report.html` in your web browser. The report groups photos into:
1. **Would be rejected — blurry:** Hard and soft motion/focus failures with reasons.
2. **Would be rejected — near-duplicate:** Redundant burst shots behind the sharpest keeper.
3. **Could not be read:** Corrupt or truncated files.
4. **Kept:** The sharpest frame from each burst and clean keepers.

### Step 3: Apply (`phototriage apply`)

Executes the reviewed plan. By default, rejects are **moved into quarantine**, never deleted outright:

```bash
# 1. Always safe: test with dry-run first
phototriage apply ~/Pictures/2026-Wedding-Smith --dry-run

# 2. Move rejects to quarantine (Default & Recommended)
phototriage apply ~/Pictures/2026-Wedding-Smith --action move

# 3. Only quarantine blurs, keep duplicates in place
phototriage apply ~/Pictures/2026-Wedding-Smith --only blurry --action move

# 4. Outright deletion (requires explicit confirmation)
phototriage apply ~/Pictures/2026-Wedding-Smith --action delete --yes
```

> **Where do rejects go?**  
> When using `--action move`, rejected files are placed in:  
> `~/Pictures/2026-Wedding-Smith/_FirstPass_rejected/blurry/`  
> `~/Pictures/2026-Wedding-Smith/_FirstPass_rejected/duplicate/`  
> Keepers remain in their original directories. If a mistake was made, simply move them back.

---

## How the AI & Scoring Signals Work

### 1. Focus Measurement (Bokeh-Preserving)
Unlike naive blur detectors that compute an average variance across the entire image, phototriage computes the Laplacian variance across **tiles** (32×32px) and takes the mean of the **sharpest 2% of tiles**.
- **Why this matters:** A portrait shot at $f/1.4$ has an ultra-sharp eye and a completely blurred background. A whole-frame detector flags it as blurry and deletes the best photo of the day. phototriage isolates the sharp subject and preserves it.

### 2. Guarded Burst Deduplication
Bursts are grouped by capture timestamp ($\le 3.0$ seconds apart) and require **both**:
1. Close 64-bit difference hash (`dHash`), **AND**
2. Normalized $16 \times 16$ grayscale shape correlation ($\ge 0.92$).
- **Why this matters:** Fine textures (like gravel, confetti, or foliage) can alias into similar perceptual hashes. The secondary shape correlation prevents unrelated textured shots from being false-flagged as duplicates.

### 3. Open-Weight CLIP AI (`openai/clip-vit-base-patch32`)
- **Model:** Open-weight `openai/clip-vit-base-patch32` (MIT License) from Hugging Face.
- **Local:** Runs locally via PyTorch; downloads once (~600 MB) to `~/.cache/huggingface/`.
- **Role:** Metadata only. CLIP is used to **label and organize** the shoot, never to decide which photo gets deleted. Deletions are strictly governed by explainable blur and burst metrics.

---

## CLI Options Reference

```bash
phototriage scan DIRECTORY [OPTIONS]
  --ai                  Enable open CLIP categorisation (default: on)
  --no-ai               Disable AI (blur and duplicate grouping only)
  --model HF_ID         Any Hugging Face CLIP checkpoint (default: openai/clip-vit-base-patch32)
  --device {auto,cpu,cuda,mps}
  --sensitivity {low,medium,high} (default: medium)
  --min-focus N         Manual focus threshold override
  --categories LIST     Comma-separated prompt categories
  --categories-file F   File with one category prompt per line
  --dedupe {hash,embedding} (embedding requires --ai)
  --no-dedupe           Disable burst deduplication
  --burst-window SECS   Time window for burst grouping (default: 3.0s)
  --jobs N              Thread count for parallel decoding (default: CPU cores)
  --out DIR             Custom directory for plan and report
  --no-thumbs           Skip thumbnail generation
  --limit N             Stop after N images

phototriage apply DIRECTORY [OPTIONS]
  --action {move,copy,delete}  Default: move to quarantine
  --dry-run             Simulate actions without changing files
  --only {blurry,duplicate}    Restrict apply to specific reject types
  --yes                 Confirmation flag required for --action delete
  --out DIR             Location of manifest if custom --out was used on scan
```

---

## Automated Test Suite

phototriage includes an in-process synthetic test suite. No real client photographs are stored or needed:

```bash
pytest tests -q
# 43 passed
```

Tests cover:
- Sharpness ordering over blurred frames
- Shallow depth-of-field / bokeh preservation
- Hash collision protection on periodic patterns
- Reversible quarantine and dry-run guarantees
- Plan versioning, file modification detection, and path escaping

---

## License

MIT License. Designed and built for Hacktoberfest Open-Source AI Challenge (Week 2: Side Hustle).
