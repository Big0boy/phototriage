# FirstPass — build plan (Hacktoberfest Open-Source AI, Week 2: Side Hustle)

> **3,000 frames come off the card. The photographer only wants to look at 400.**
> A wedding and event shooter points FirstPass at the folder, then everything runs on their own
> machine: no upload, no per-image bill, no client photos leaving the drive.

Submission due **Sun Oct 18, 2026, 11:59 PM PDT**. First-user bonus requires shipping **Sat Oct 17**.

---

> **Status: the shipping implementation is the Python CLI in this directory.** See `README.md`
> and `phototriage.py`; run it with `python phototriage.py scan DIR --ai`. The browser build
> described in §4–5 was replaced because a desktop tool is the right shape for a job that
> starts with a folder of 3,000 JPEGs on a photographer's own machine. §2 (the open-vs-closed
> thesis), §3 (scope and the stated boundaries), §8 (first-user plan) and §9 (the numbers the
> post needs) all still apply. §5's performance gate is now answered: 0.052 s/image on CPU, so
> the corpus-wide scoring branch is the one to build.

---

## 1. Who pays, and for what

**Buyer:** solo wedding / event photographers, 20–40 shoots a year, 2,000–4,000 frames each.

**The pain:** culling is 4–8 hours per wedding of eyeballing near-identical burst frames to find
the one where nobody blinked. It's the single most-hated unpaid task in the job, and it happens
after the adrenaline of the shoot, on a deadline.

**What they already do:** pay **$20–40/month** for Aftershoot, Narrative Select, or Imagen — all
cloud-based. That's the willingness-to-pay proof. **Verify current pricing on the day** and quote
the real numbers in the writeup; the whole business case rests on them.

**What we sell:** a fast first pass. Not "AI picks your best photos" (a promise no model keeps) but
"here are the 400 frames with a fighting chance, ranked inside each burst, and here's why each
reject was rejected." The photographer stays the editor. That honesty is also what makes it
shippable in three days.

---

## 2. Why open is load-bearing (the thesis of the DEV post)

This is the part that wins or loses the challenge, so it goes in the README as the headline.

1. **The data legally can't go to a closed API.** Wedding couples sign contracts; a growing number
   of them say the photos won't be fed to third-party AI services. Every competitor above is a
   closed pipeline, and the photographer has no way to audit what happens to 3,000 frames of
   someone's family. Local inference isn't a cost optimisation here — it's the only version that
   clears the contract. A testimonial-shaped line for the post: *"I can put this in my contract."*
2. **Marginal cost per frame is zero.** 3,000 frames × 30 weddings = **90,000 images a year**. On
   per-image cloud pricing, that's the subscription *and* the margin. Locally it's electricity.
   That's what makes a $15/mo price possible at all.
3. **The model is the swappable part.** A photographer who shoots 80% indoor receptions or a
   motorsport specialist can re-weight or swap the scoring prompts/model without asking anyone's
   permission. That's a fine-tune path no closed vendor can offer.
4. **It works on a plane, in a cabin, in a hotel with awful wifi** — which is exactly where
   photographers do their culling, before they fly home.

---

## 3. Scope

**In:**
- Chromium desktop (Chrome/Edge). `showDirectoryPicker()` → read a folder in place.
- JPEG only. 2,000–4,000 frames is the target batch.
- Per-frame signals: **technical failure** (soft, motion-blurred, badly underexposed), **blink
  risk**, **quality/aesthetic score**, **burst membership**.
- Keyboard-driven triage UI: `K` keep / `X` reject / `→` next, with the reason for each verdict
  shown next to the frame and the burst laid out as a filmstrip with the recommended frame marked.
- Output: a rejects CSV + XMP rating sidecars + optional move to `_FirstPass_rejects/`.

**Out (say this out loud in the post — judges respect a stated boundary):**
- RAW. It's a different decoding problem and would eat the whole budget.
- Any claim of taste. We do not tell anyone which of two good frames is "better"; we say which one
  is technically alive.
- Non-Chromium browsers (no folder access). File-input fallback only if time allows.
- Face recognition, identity grouping, any inference about *who* is in the frame.

---

## 4. Architecture

Everything in-browser, zero backend, replicating the LeafPace shape. Vite + React + TypeScript +
Tailwind PWA, service worker so the app itself is installable and offline after first load.

### Pipeline (cheap → expensive, so a dead frame costs almost nothing)

| # | Stage | What runs | Cost |
|---|-------|-----------|------|
| 1 | Enumerate | `showDirectoryPicker()` + `for await` handles | ~0 |
| 2 | Metadata | EXIF `DateTimeOriginal`, orientation (`exifr`, open) | ms |
| 3 | Decode-small | `createImageBitmap(file, { resizeWidth: 512, imageOrientation: 'from-image' })` in a Worker | browser-native |
| 4 | Blur gate | Laplacian variance, pure JS on grayscale 512px | ~5 ms |
| 5 | Face/blink | MediaPipe `FaceLandmarker` (`@mediapipe/tasks-vision`, Apache-2.0) → `eyeBlinkLeft`/`eyeBlinkRight` blendshapes; no face detected is a *pass*, not a reject | ~10–20 ms |
| 6 | Quality score | **CLIP** (`Xenova/clip-vit-base-patch32` via transformers.js) zero-shot against text prompts — the CLIP-IQA technique (Wang et al., AAAI 2023). Also emits the image embedding | ~20–40 ms |
| 7 | Burst clustering | EXIF time proximity + cosine similarity of CLIP embeddings (union-find, pure JS) | ~0 |
| 8 | Rank | Per burst, order by quality score, mark the top frame "recommended" | ~0 |
| 9 | Verdict | Reject = hard fail (blur / blink / exposure) or not the burst winner **and** scored below threshold. Every reject carries its reason string. | ~0 |

**One model, two jobs.** CLIP supplies both the quality score and the embeddings that make burst
dedupe work. That's deliberate: one open-weight download (~150 MB), one warm-up, and a single
consistent notion of "similar."

**Why CLIP-IQA rather than an ONNX-trained scorer:** text prompts are inspectable and editable, so
the model is genuinely swappable/tunable — which is argument #3 in the post, made real. Fallbacks
if the spike says CLIP is too slow or too blunt, in order: (a) `onnxruntime-web` + a MUSIQ/NIMA-class
open IQA checkpoint for the technical score only, (b) a purpose-trained open no-reference IQA model,
(c) drop the aesthetic score to the blur gate alone and be honest in the post that we triage
technical failures, not taste.

### Non-negotiables carried over from week 1

- **Never blocks.** The grid of thumbnails paints from stage 1–3 immediately; scores stream in.
  If WebGPU is unavailable, degrade to the WASM backend and a lower resolution, don't die.
- **Show the reason.** Every verdict is explainable in one sentence. This is the product's
  credibility.
- **Model choice lives behind one constant** in a single module, like `webgpu-llm.ts` in week 1.

### Layout

```
week2/
  src/
    components/     TriageGrid, BurstFilmstrip, DropZone, ProgressBar, VerdictBadge
    services/
      folder.ts     File System Access + directory enumeration + fallback
      decode.ts     Worker: createImageBitmap downscale, orientation, grayscale
      blur.ts       Laplacian variance
      faces.ts      MediaPipe FaceLandmarker wrapper (blink chance)
      quality.ts    transformers.js CLIP scoring + embeddings
      bursts.ts     time+embedding union-find clustering
      triage.ts     the cascade + verdict + reason strings
      export.ts     rejects CSV, XMP sidecars, optional move
    workers/pipeline.worker.ts
    __tests__/      pure-logic tests (see below)
  scripts/
    smoke.mjs       CDP headless run over a synthetic 40-frame fixture set
    bench.mjs       throughput benchmark — the day-1 gate
  public/fixtures/  generated synthetic burst set (no real client photos, ever)
```

### Tests (week-1 discipline, same shape)

Vitest over the pure logic — blur scoring on synthetic sharp/blurred pairs, burst clustering given
fabricated timestamps + embeddings, ranking, verdict/reason generation, CSV and XMP serialisation,
EXIF time parsing. Plus the dependency-free CDP smoke test driving the real app over a fixture
folder, and `bench.mjs` as the performance gate. CI on every push.

---

## 5. Performance budget and the day-1 gate

Target: **3,000 frames in under 5 minutes** on a laptop with a discrete GPU, under 15 on integrated.

This is the number that decides the project, so measure it before building any UI. **Spike, day 1:**

1. Generate 300 synthetic JPEGs (~6 MB, mixed sharp/blurred) with a script. Never test on anyone's
   real photos.
2. Measure, in a Worker: enumerate → decode-small → blur → face → CLIP, reporting per-stage ms and
   peak memory.
3. Extrapolate to 3,000 and decide:
   - **< 0.1 s/frame** → build the full aesthetic score.
   - **0.1–0.3 s/frame** → keep CLIP but run it one resolution step down (384px) and consider a
     cheap pre-filter that skips CLIP entirely for already-rejected frames.
   - **> 0.3 s/frame** → CLIP only on burst winners + frames that passed stages 4–5. Aesthetic
     ranking becomes "best of burst" rather than corpus-wide.
4. Whatever the outcome, **write the measured number down** and put it in the post. A real
   throughput figure on real hardware is worth more than any adjective.

Memory is the quiet killer: 3,000 bitmaps at 512px must be streamed and released, never held.
Thumbnails get rendered to a canvas strip / `ImageBitmap` transferred and closed as the grid
scrolls.

---

## 6. Day plan

**Thu (tonight)** — repo scaffold, fixtures generator, `bench.mjs`, run the spike. Decide the
architecture branch from §5. Do not start the UI before this number exists.

**Fri** — the cascade end to end on a batch (no UI polish), burst clustering, verdicts + reasons,
export (CSV + XMP). Vitest green on the pure logic.

**Sat** — triage UI (grid, keyboard flow, filmstrip, reason badges), progress + never-block
behaviour, service worker. **Ship it publicly by evening** and hand it to a photographer.

**Sun** — act on whatever the first user says (even if it's "it flagged the wrong frame"), write
the DEV post, submit before 11:59 PM PDT. Record the 60-second demo GIF: 3,000 frames → 400 at
~10× visible speed.

Contingency: if the spike lands badly and Friday goes sideways, fall back to **"blur + blink +
bursts only, no aesthetic score."** That's still a real tool that saves hours, still only possible
locally, and still a complete post. Ship that rather than nothing.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Throughput misses the budget | The three-tier fallback in §5, decided by measurement, not hope |
| Blink detection false-positives ruin trust | Never auto-reject on blink alone — badge it as "possible blink" and let the human decide |
| CLIP aesthetic score feels arbitrary | Present it as *ranking within a burst*, never as an absolute judgement; show the prompt used |
| File System Access quirks | Chromium-only, stated; `webkitdirectory` input as a fallback path |
| No photographer gets back to you in 24h | Line up **two** people before Saturday, and be willing to sit in on their cull |
| Temptation to keep adding | The scope list in §3 is the contract. Anything outside it goes in a "next week" section of the post |

---

## 8. First-user plan (the bonus points)

Find photographers at: local wedding-photographer directories, a regional photographers' Facebook
group, r/AskPhotography and r/WeddingPhotography (check each sub's self-promo rules first), or a
neighbour who shoots events.

Ask for **twenty minutes and one real culled-but-not-yet-delivered shoot**, then watch them use it
without helping. The post needs their words, not our adjectives:

- How long does one wedding take you to cull today?
- Would you have paid for this? What would it be worth per month?
- What did it flag that you'd have kept? *(Most useful answer — put it in the post, and in the
  "what's next" section.)*
- Would a local-only guarantee change what you can promise a couple in your contract?

If they say they'd pay, say the price out loud and see what happens. That's a better story than
any survey.

---

## 9. The numbers the post needs (fill in as measured)

| Claim | Value | Source |
|---|---|---|
| Frames in a typical shoot | 2,000–4,000 | ask the first user |
| Hours of culling saved | | first user's own number |
| FirstPass throughput, 3,000 frames | ms/frame, total | `bench.mjs` on named hardware |
| Cloud competitor price per image / month | | their pricing page, verified |
| Annual per-image volume | 90,000 images | 3,000 × 30 shoots |
| Marginal cost per frame, local | ~0 | by construction |
| Model size downloaded once | ~150 MB, cached | measured |
| Bytes uploaded to any server | **0** | by architecture |

---

## 10. Reuse from LeafPace (week 1)

| LeafPace piece | Reused as |
|---|---|
| Vite/React/TS PWA + service worker | the whole shell |
| Zero-backend + static-host deploy (GitHub Pages) | hosting, $0 |
| Model-behind-one-constant pattern (`webgpu-llm.ts`) | `quality.ts` |
| Never-blocks + deterministic fallback | grid paints before scores arrive |
| Dependency-free CDP smoke harness | `scripts/smoke.mjs` over a fixture folder |
| Vitest suite shape | `src/__tests__/` |
| The open-vs-closed writeup structure | §2 of the DEV post |

Carrying these over is what makes a three-day build possible; say so in the post, and link week 1.
