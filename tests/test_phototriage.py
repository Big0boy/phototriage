"""Synthetic fixtures and tests for phototriage.

Every fixture is generated in-process: this repository never contains a real photograph,
and the tests never need one. The image constants below are deliberately extreme (a
radius-12 gaussian blur against a random-noise texture) so the assertions test the
contract — that sharp orders above blurred, that bokeh is spared, that only rejects are
touched — rather than the tuning of the default thresholds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import phototriage as pt  # noqa: E402
from phototriage import Record, VERDICT_BLURRY, VERDICT_DUPLICATE, VERDICT_KEEP  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "phototriage.py"

# A valid, non-constant 16x16 signature: identical signatures correlate at 1.0.
SIG = bytes([0, 255] * (pt.SIGNATURE_SIZE * pt.SIGNATURE_SIZE // 2)).hex()


# --------------------------------------------------------------------------------------
# fixture builders
# --------------------------------------------------------------------------------------


def texture(size=(1200, 800), seed=7):
    """High-frequency random texture: unambiguously sharp."""
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype="uint8")
    return Image.fromarray(arr, "RGB")


def scene(size=(1200, 800), seed=7, blur_radius=0.0):
    """A smooth synthetic scene: a seed-specific low-frequency background plus blocks and
    edges, closer to a photograph.

    The background must vary with the seed. An earlier version gave every scene the same
    fixed gradient, which made unrelated fixtures agree on 8 of 64 hash bits and correlate
    at 0.97 — the fixtures were accidentally near-duplicates of each other, and the tests
    were validating a detector against images no camera produces.
    """
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np

    rng = np.random.default_rng(seed)
    coarse = rng.integers(25, 220, (5, 7, 3), dtype="uint8")
    img = Image.fromarray(coarse, "RGB").resize(size, Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(img)
    for _ in range(14):
        x, y = int(rng.integers(0, size[0])), int(rng.integers(0, size[1]))
        w, h = int(rng.integers(60, 260)), int(rng.integers(60, 260))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=3)
        draw.ellipse([x, y, x + w // 2, y + h // 2], fill=colour)
    if blur_radius:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return img


def flush_bokeh(size=(1200, 800), seed=3, blur=24, period=8):
    """A sharp subject against a heavily defocused background — the frame a global blur
    detector throws away: sharp texture in the centre, blurred surround."""
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np

    rng = np.random.default_rng(seed)
    background = Image.fromarray(
        rng.integers(20, 70, (size[1], size[0], 3), dtype="uint8"), "RGB"
    ).filter(ImageFilter.GaussianBlur(blur))

    centre_w, centre_h = size[0] // 3, size[1] // 3
    centre = Image.new("RGB", (centre_w, centre_h))
    draw = ImageDraw.Draw(centre)
    for y in range(0, centre_h, period):
        for x in range(0, centre_w, period):
            if (x // period + y // period) % 2 == 0:
                draw.rectangle([x, y, x + period - 1, y + period - 1], fill=(255, 255, 255))

    canvas = background.copy()
    canvas.paste(centre, ((size[0] - centre_w) // 2, (size[1] - centre_h) // 2))
    return canvas


def write_set(directory: Path, count=3, blur=True, prefix="frame", **kwargs):
    """Write a small, deterministic batch of JPEGs."""
    made = []
    for index in range(count):
        img = scene(seed=index, **kwargs) if not blur else scene(seed=index, blur_radius=12)
        path = directory / f"{prefix}_{index:02d}.jpg"
        img.save(path, "JPEG", quality=92)
        made.append(path)
    return made


def run_cli(*args, cwd=None):
    """Run the CLI as a subprocess — the interface a user actually has."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *[str(a) for a in args]],
        capture_output=True, text=True, cwd=cwd, timeout=300,
    )


# --------------------------------------------------------------------------------------
# focus measurement
# --------------------------------------------------------------------------------------


def test_sharp_orders_above_blurred():
    import numpy as np
    from PIL import Image

    sharp = np.asarray(texture().convert("L").resize((1024, 683), Image.Resampling.BOX))
    blurred = np.asarray(
        scene(size=(1024, 683), blur_radius=12).convert("L")
    )
    sharp_focus, _, _ = pt.focus_scores(sharp)
    blur_focus, _, _ = pt.focus_scores(blurred)
    assert sharp_focus > blur_focus * 5, f"{sharp_focus=} {blur_focus=}"


def test_bokeh_background_does_not_mask_a_sharp_subject():
    """The whole reason for measuring tiles instead of the frame. A 90th-percentile rule
    failed this frame: its sharp subject covers about a third of the tiles, and with a
    small subject it is missed entirely. Measured here at 31k-42k against 3 for a hard blur."""
    import numpy as np
    from PIL import Image

    bokeh = np.asarray(flush_bokeh().convert("L").resize((1024, 683), Image.Resampling.BOX))
    blurred = np.asarray(
        scene(size=(1024, 683), blur_radius=14).convert("L")
    )
    focus, _, median = pt.focus_scores(bokeh)
    blur_focus, _, _ = pt.focus_scores(blurred)
    assert focus > median, "a sharp region must stand above the typical tile"
    assert focus > blur_focus * 100, f"shallow depth of field must not read as blur: {focus=}"


def test_small_off_centre_subject_survives_the_blur_threshold():
    """A subject covering a few percent of the frame — where a percentile rule gave up."""
    import numpy as np
    from PIL import Image

    small = np.asarray(
        flush_bokeh(size=(1200, 800), seed=2, period=7).convert("L")
        .resize((1024, 683), Image.Resampling.BOX)
    )
    focus, _, _ = pt.focus_scores(small)
    assert focus > 1000, f"a small sharp subject scored {focus:.1f}"


def test_projected_embedding_accepts_both_transformers_api_shapes():
    """4.x returned a tensor, 5.x returns a ModelOutput — both must work."""
    torch = pytest.importorskip("torch")

    class FakeOutput:
        def __init__(self, tensor):
            self.pooler_output = tensor

    assert pt.projected_embedding(torch.zeros(2, 512), 512).shape == (2, 512)
    assert pt.projected_embedding(FakeOutput(torch.zeros(2, 512)), 512).shape == (2, 512)


def test_projected_embedding_refuses_an_unprojected_output():
    """Comparing unprojected vision features against projected text features would be
    silently meaningless, so it has to be an error rather than a wrong answer."""
    torch = pytest.importorskip("torch")

    class Unprojected:
        pooler_output = None

    class FakeOutput:
        def __init__(self, tensor):
            self.pooler_output = tensor

    unprojected = FakeOutput(torch.zeros(8, 50, 768))
    with pytest.raises(RuntimeError, match="would not be comparable"):
        pt.projected_embedding(unprojected, 512)

    empty = FakeOutput(None)
    with pytest.raises(RuntimeError, match="no embedding tensor"):
        pt.projected_embedding(empty, 512)
    assert Unprojected().pooler_output is None


def stripes(period, phase, size=(1200, 800)):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for x in range(-period, size[0] + period, period * 2):
        draw.rectangle([x + phase, 0, x + phase + period - 1, size[1]], fill=(235, 235, 235))
    return img


def test_fine_texture_aliases_to_a_matching_hash_and_the_guard_splits_it():
    """The real degeneracy, measured on this fixture: at 9x8 a fine periodic texture is
    sampled below its own frequency, so two frames of it at different phases hash eight
    bits apart (inside the grouping distance) while correlating at only 0.38. Without the
    second signal those are "duplicates" of each other and a real photo gets deleted."""
    a, b = stripes(8, 0), stripes(8, 3)
    assert pt.hamming(pt.dhash(a), pt.dhash(b)) <= pt.HASH_MAX_DISTANCE
    assert pt.correlate(pt.signature(a), pt.signature(b)) < pt.SIGNATURE_MIN_CORRELATION

    records = make_records([100, 101])
    records[0].taken_at, records[1].taken_at = 1000.0, 1000.5
    records[0].dhash, records[1].dhash = pt.dhash(a), pt.dhash(b)
    records[0].sig, records[1].sig = pt.signature(a), pt.signature(b)
    pt.group_bursts(records, 3.0, "hash")
    assert all(r.group_size == 1 for r in records)


def test_unrelated_frames_are_not_grouped_as_a_burst():
    records = make_records([100, 101])
    a, b = scene(seed=1), scene(seed=2)
    records[0].taken_at, records[1].taken_at = 1000.0, 1000.5
    records[0].dhash, records[1].dhash = pt.dhash(a), pt.dhash(b)
    records[0].sig, records[1].sig = pt.signature(a), pt.signature(b)
    pt.group_bursts(records, 3.0, "hash")
    assert all(r.group_size == 1 for r in records)


def test_frames_of_the_same_shot_are_grouped():
    records = make_records([100, 101, 102])
    frames = [scene(seed=5) for _ in range(3)]
    for index, (rec, frame) in enumerate(zip(records, frames)):
        rec.taken_at = 1000.0 + index * 0.3
        rec.dhash, rec.sig = pt.dhash(frame), pt.signature(frame)
    pt.group_bursts(records, 3.0, "hash")
    assert {r.group_size for r in records} == {3}


def test_signature_still_matches_for_frames_of_the_same_shot():
    """Exposure drift must not break the match — mean-centring and span normalisation
    make the fingerprint insensitive to it by design."""
    from PIL import ImageEnhance

    base = scene(seed=3)
    darker = ImageEnhance.Brightness(base).enhance(0.8)
    assert pt.correlate(pt.signature(base), pt.signature(darker)) >= pt.SIGNATURE_MIN_CORRELATION
    assert len(pt.signature(base)) == pt.SIGNATURE_SIZE * pt.SIGNATURE_SIZE * 2


def test_grouping_needs_corroborating_shape_not_just_a_matching_hash():
    records = make_records([100, 101])
    records[0].taken_at, records[1].taken_at = 1000.0, 1000.5
    records[0].dhash = records[1].dhash = "0" * 16
    records[0].sig = SIG
    records[1].sig = bytes([0] * 128 + [255] * 128).hex()  # identical hash, different shape
    pt.group_bursts(records, 3.0, "hash")
    assert all(r.group_size == 1 for r in records), "matching hash alone must not group frames"


def test_correlation_of_unusable_signatures_is_zero():
    assert pt.correlate("", SIG) == 0.0
    assert pt.correlate("zz", SIG) == 0.0


def test_hamming_distance():
    assert pt.hamming("0" * 16, "0" * 16) == 0
    assert pt.hamming("0" * 16, "f" * 16) == 64
    assert pt.hamming("0000000000000000", "0000000000000001") == 1


def test_dhash_is_stable_across_brightness_and_orientation_of_content():
    from PIL import Image, ImageEnhance

    base = scene(seed=11)
    brighter = ImageEnhance.Brightness(base).enhance(1.25)
    different = scene(seed=99)
    assert pt.hamming(pt.dhash(base), pt.dhash(brighter)) <= 6
    assert pt.hamming(pt.dhash(base), pt.dhash(different)) > 6
    assert isinstance(Image.new("RGB", (10, 10)), Image.Image)


# --------------------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------------------


def make_records(focus_values, start=1000.0):
    records = []
    for index, focus in enumerate(focus_values):
        records.append(Record(
            path=f"frame_{index}.jpg", name=f"frame_{index}.jpg",
            size=1000, mtime=start + index, taken_at=start + index * 60,
            focus=float(focus), focus_median=float(focus) / 2.0, dhash="0" * 16, sig=SIG,
        ))
    return records


def test_relative_threshold_flags_only_the_outliers():
    """A soft shoot pulls its own threshold down with it — this is the relative branch."""
    records = make_records([100, 200, 3000, 4000])
    meta, median = pt.decide(records, "medium", None, "off")
    assert median == 3000.0
    assert meta.threshold == max(25.0, 0.08 * 3000.0) == 240.0
    assert [r.verdict for r in records] == [VERDICT_BLURRY, VERDICT_BLURRY, VERDICT_KEEP, VERDICT_KEEP]
    assert "shoot median" in records[0].reason


def test_a_batch_of_keepers_is_left_alone():
    """The common case: a consistent shoot must come back with no rejections at all."""
    records = make_records([2000, 2400, 3000, 3400])
    meta, _ = pt.decide(records, "medium", None, "off")
    assert meta.threshold == 240.0
    assert all(r.verdict == VERDICT_KEEP for r in records)


def test_uniformly_bad_batch_still_flagged_by_the_absolute_floor():
    """If every frame is soft, a purely relative rule would forgive all of them."""
    records = make_records([5, 6, 7, 8])
    meta, _ = pt.decide(records, "medium", None, "off")
    assert meta.threshold == pt.SENSITIVITY["medium"][1] == 25.0
    assert all(r.verdict == VERDICT_BLURRY for r in records)


def test_sensitivity_presets_order_the_threshold():
    low, _ = pt.decide(make_records([2000, 3000, 4000]), "low", None, "off")
    med, _ = pt.decide(make_records([2000, 3000, 4000]), "medium", None, "off")
    high, _ = pt.decide(make_records([2000, 3000, 4000]), "high", None, "off")
    assert low.threshold < med.threshold < high.threshold
    assert (low.threshold, med.threshold, high.threshold) == (120.0, 240.0, 450.0)


def test_min_focus_override_replaces_the_preset():
    records = make_records([80, 400])
    meta, _ = pt.decide(records, "medium", 100.0, "off")
    assert meta.threshold == 100.0
    assert [r.verdict for r in records] == [VERDICT_BLURRY, VERDICT_KEEP]


def test_burst_keeps_the_sharpest_and_rejects_the_rest():
    records = make_records([120, 300, 200])
    for index, rec in enumerate(records):
        rec.taken_at = 1000.0 + index * 0.4
        rec.dhash = "0" * 16
    pt.group_bursts(records, 3.0, "hash")
    assert {r.group_size for r in records} == {3}

    meta, _ = pt.decide(records, "medium", None, "hash")
    assert records[1].verdict == VERDICT_KEEP
    assert records[0].verdict == VERDICT_DUPLICATE
    assert records[2].verdict == VERDICT_DUPLICATE
    assert "near-duplicate of frame_1.jpg" in records[0].reason


def test_frames_far_apart_in_time_are_never_grouped():
    records = make_records([100, 400])
    records[0].taken_at, records[1].taken_at = 1000.0, 1040.0
    pt.group_bursts(records, 3.0, "hash")
    assert all(r.group_size == 1 for r in records)


def test_a_blurry_frame_is_never_reported_as_a_duplicate():
    """Blur is the more actionable reason; it must win the label."""
    records = make_records([1, 300])
    records[0].taken_at, records[1].taken_at = 1000.0, 1000.5
    records[0].dhash = records[1].dhash = "0" * 16
    pt.group_bursts(records, 3.0, "hash")
    pt.decide(records, "medium", None, "hash")
    assert records[0].verdict == VERDICT_BLURRY


# --------------------------------------------------------------------------------------
# scan outputs
# --------------------------------------------------------------------------------------


@pytest.fixture
def shoot(tmp_path):
    root = tmp_path / "shoot"
    root.mkdir()
    # two sharp frames seconds apart, near-identical -> a burst
    scene(seed=1).save(root / "burst_a.jpg", "JPEG", quality=92)
    scene(seed=1).save(root / "burst_b.jpg", "JPEG", quality=92)
    # a hard blur, a mild blur, a keeper, and a name that needs escaping
    scene(seed=2, blur_radius=14).save(root / "blurry.jpg", "JPEG", quality=92)
    texture().save(root / "sharp.jpg", "JPEG", quality=92)
    texture(seed=5).save(root / "a&b <tag>.jpg", "JPEG", quality=92)
    (root / "notes.txt").write_text("not an image")
    return root


def test_scan_writes_plan_and_touches_nothing(shoot):
    before = {p.name: p.stat().st_mtime for p in shoot.iterdir()}
    result = run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    assert result.returncode == 0, result.stderr

    out = shoot / pt.OUT_DIR_NAME
    manifest = json.loads((out / pt.MANIFEST_NAME).read_text())
    assert manifest["meta"]["schema_version"] == pt.SCHEMA_VERSION
    assert (out / "triage.csv").exists()
    assert (out / "report.html").exists()

    names = {r["name"] for r in manifest["records"]}
    assert "notes.txt" not in names, "non-images must be ignored"
    assert "blurry.jpg" in names

    after = {p.name: p.stat().st_mtime for p in shoot.iterdir() if p.is_file()}
    for name, mtime in before.items():
        assert after.get(name) == mtime, f"{name} was modified by scan"
    assert not (shoot / pt.QUARANTINE_DIR).exists()


def test_scan_flags_the_hard_blur_and_keeps_the_sharp_frames(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high", "--no-dedupe")
    manifest = json.loads((shoot / pt.OUT_DIR_NAME / pt.MANIFEST_NAME).read_text())
    by_name = {r["name"]: r for r in manifest["records"]}
    assert by_name["blurry.jpg"]["verdict"] == VERDICT_BLURRY
    assert by_name["sharp.jpg"]["verdict"] == VERDICT_KEEP
    assert by_name["blurry.jpg"]["focus_p90"] < by_name["sharp.jpg"]["focus_p90"]


def test_report_escapes_filenames(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs")
    html = (shoot / pt.OUT_DIR_NAME / "report.html").read_text()
    assert "a&amp;b &lt;tag&gt;.jpg" in html
    assert "a&b <tag>.jpg" not in html


def test_scan_is_idempotent_and_skips_its_own_output(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs")
    run_cli("scan", shoot, "--no-ai", "--no-thumbs")
    manifest = json.loads((shoot / pt.OUT_DIR_NAME / pt.MANIFEST_NAME).read_text())
    assert all(not r["path"].startswith(pt.OUT_DIR_NAME) for r in manifest["records"])


# --------------------------------------------------------------------------------------
# apply: the only phase allowed to change the filesystem
# --------------------------------------------------------------------------------------


def test_apply_move_quarantines_rejects_only(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    result = run_cli("apply", shoot, "--action", "move")
    assert result.returncode == 0, result.stderr

    quarantine = shoot / pt.QUARANTINE_DIR / "blurry"
    quarantined = {p.name for p in quarantine.iterdir()}
    assert "blurry.jpg" in quarantined
    assert not (shoot / "blurry.jpg").exists(), "the reject should have moved"
    assert (shoot / "sharp.jpg").exists(), "a sharp frame was moved but should have been kept"

    # burst_a/burst_b are the same frame twice: exactly one must survive. Which one is
    # a tie-break, and the contract is the count, not the name.
    survivors = [n for n in ("burst_a.jpg", "burst_b.jpg") if (shoot / n).exists()]
    assert len(survivors) == 1, f"a burst of two identical frames should keep one, kept {survivors}"
    duplicated = {p.name for p in (shoot / pt.QUARANTINE_DIR / "duplicate").iterdir()}
    assert len(duplicated) == 1 and duplicated.pop() not in survivors


def test_apply_copy_leaves_the_original_in_place(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    result = run_cli("apply", shoot, "--action", "copy")
    assert result.returncode == 0, result.stderr
    assert (shoot / "blurry.jpg").exists(), "copy must never remove the original"
    copied = {p.name for p in (shoot / pt.QUARANTINE_DIR / "blurry").iterdir()}
    assert "blurry.jpg" in copied
    assert (shoot / "sharp.jpg").exists()


def test_limit_scans_only_the_first_n_frames(shoot):
    result = run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--limit", "2")
    assert result.returncode == 0, result.stderr
    manifest = json.loads((shoot / pt.OUT_DIR_NAME / pt.MANIFEST_NAME).read_text())
    assert len(manifest["records"]) == 2


def test_apply_delete_refuses_without_yes(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    result = run_cli("apply", shoot, "--action", "delete")
    assert result.returncode == 2
    assert "--yes" in result.stderr
    assert (shoot / "blurry.jpg").exists(), "refused run must not delete anything"


def test_apply_delete_with_yes_removes_only_rejects(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    result = run_cli("apply", shoot, "--action", "delete", "--yes")
    assert result.returncode == 0, result.stderr
    assert not (shoot / "blurry.jpg").exists()
    assert (shoot / "sharp.jpg").exists()


def test_apply_dry_run_changes_nothing(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    result = run_cli("apply", shoot, "--dry-run")
    assert result.returncode == 0
    assert (shoot / "blurry.jpg").exists()
    assert not (shoot / pt.QUARANTINE_DIR).exists()


def test_apply_skips_files_modified_after_the_scan(shoot):
    """A stale plan is a plan about different files. Refuse rather than guess."""
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    target = shoot / "blurry.jpg"
    time.sleep(0.02)
    target.write_bytes(b"x" * 20_000)          # size and mtime both change
    os.utime(target, (time.time() + 5, time.time() + 5))

    result = run_cli("apply", shoot, "--action", "move")
    assert "skip (modified since scan)" in result.stdout
    assert target.exists()


def test_apply_only_filter(shoot):
    run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high")
    result = run_cli("apply", shoot, "--action", "move", "--only", "duplicate")
    assert result.returncode == 0
    assert (shoot / "blurry.jpg").exists(), "--only duplicate must leave blur alone"


def test_apply_reads_a_plan_written_to_a_custom_out(shoot, tmp_path):
    """scan --out must be paired with apply --out; the README documents exactly this."""
    plan = tmp_path / "plan"
    result = run_cli("scan", shoot, "--no-ai", "--no-thumbs", "--sensitivity", "high", "--out", plan)
    assert result.returncode == 0, result.stderr
    assert (plan / pt.MANIFEST_NAME).exists()
    assert not (shoot / pt.OUT_DIR_NAME).exists(), "--out must keep the plan out of the shoot"

    # without --out, apply cannot find the plan and must refuse rather than guess
    refused = run_cli("apply", shoot, "--dry-run")
    assert refused.returncode == 2
    assert "run scan first" in refused.stderr

    # with the matching --out, the plan is applied and only rejects move
    applied = run_cli("apply", shoot, "--out", plan, "--action", "move")
    assert applied.returncode == 0, applied.stderr
    assert not (shoot / "blurry.jpg").exists()
    assert (shoot / "sharp.jpg").exists()


def test_apply_without_a_manifest_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_cli("apply", empty)
    assert result.returncode == 2
    assert "run scan first" in result.stderr


def test_refuses_to_operate_on_filesystem_root():
    result = run_cli("scan", "/", "--no-ai", "--limit", "1")
    assert result.returncode == 2
    assert "refusing" in result.stderr


def test_no_images_is_a_clean_failure(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    result = run_cli("scan", empty, "--no-ai")
    assert result.returncode == 1
    assert "no images found" in result.stdout
