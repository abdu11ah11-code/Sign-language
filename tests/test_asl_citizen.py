"""Tests for the ASL Citizen importer.

A miniature dataset is built on disk — real .mp4 files and real split CSVs,
zipped the way the release ships — so the whole path is exercised: reading
members out of the zip, matching the split files' columns, tracking, sharding,
the manifest and resuming.

MediaPipe itself is replaced by a stand-in tracker: these tests check the
plumbing, not MediaPipe's detection quality.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sign_bridge.capture.holistic import RAW_DIM
from sign_bridge.data import asl_citizen as ac

#: The real factory, kept before the autouse fixture replaces it.
REAL_HOLISTIC = ac._holistic

SIGNS = ["mom", "dad", "drink"]
PARTICIPANTS = ["P01", "P02"]


def _write_video(path: Path, frames: int = 12) -> None:
    """A small real video file (noise), so OpenCV genuinely has to read it."""
    cv2 = pytest.importorskip("cv2")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    rng = np.random.default_rng(0)
    try:
        for _ in range(frames):
            writer.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    finally:
        writer.release()


def _mini_dataset(tmp_path: Path, as_zip: bool = True) -> Path:
    """Build a small ASL-Citizen-shaped dataset; return the zip or the folder."""
    root = tmp_path / "ASL_Citizen"
    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for sign in SIGNS:
        for p_index, participant in enumerate(PARTICIPANTS):
            for take in range(2):
                name = f"{participant}-{sign}-{take}.mp4"
                _write_video(root / "videos" / name)
                split = ("train", "val", "test")[(p_index * 2 + take) % 3]
                rows[split].append({"Participant ID": participant, "Video file": name,
                                    "Gloss": sign.upper(), "ASL-LEX Code": f"{sign}1"})
    for split, items in rows.items():
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=["Participant ID", "Video file", "Gloss",
                                                 "ASL-LEX Code"])
        writer.writeheader()
        writer.writerows(items)
        (root / "splits").mkdir(parents=True, exist_ok=True)
        (root / "splits" / f"{split}.csv").write_text(out.getvalue(), encoding="utf-8")
    if not as_zip:
        return root
    archive = tmp_path / "ASL_Citizen.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for item in sorted(root.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(tmp_path))
    return archive


class _FakeTracker:
    """Stands in for MediaPipe: a hand present in all but the first frame.

    Args:
        trim: Hold back the hand on the first frame, so the trimming of the
            still moments before a sign gets exercised. Off for the threaded
            test, where "first frame" would mean the first of each *thread's*
            videos and so depend on how the work happened to be shared out.
    """

    def __init__(self, trim: bool = True) -> None:
        self.trim = trim
        self.calls = 0
        self.closed = False
        self._lock = threading.Lock()

    def process(self, rgb: np.ndarray) -> Any:
        with self._lock:
            self.calls += 1
            return self.calls if self.trim else 2

    def close(self) -> None:
        self.closed = True


def _fake_features(result: Any) -> np.ndarray:
    frame = np.zeros(RAW_DIM, np.float32)
    if int(result) > 1:                      # no hand in the very first frame
        frame[:126] = 0.5
        frame[126:162] = 0.4
    return frame


@pytest.fixture(autouse=True)
def _no_mediapipe(monkeypatch: pytest.MonkeyPatch) -> _FakeTracker:
    tracker = _FakeTracker()
    monkeypatch.setattr(ac, "_holistic", lambda: tracker)
    monkeypatch.setattr("sign_bridge.capture.holistic.extract_features", _fake_features)
    return tracker


# ── gloss matching ───────────────────────────────────────────────────────
@pytest.mark.parametrize(("raw", "expected"), [
    ("ice cream", "ICECREAM"), ("ICE_CREAM", "ICECREAM"), ("ice-cream1", "ICECREAM"),
    ("  Mom  ", "MOM"), ("thank you", "THANKYOU"), ("TV", "TV"),
])
def test_normalize_gloss(raw: str, expected: str) -> None:
    assert ac.normalize_gloss(raw) == expected


def test_column_detection_is_tolerant() -> None:
    csv_text = ("Participant ID,Video file,Gloss\nP01,a.mp4,MOM\n")
    rows = ac.read_index(csv_text, "train", {"a.mp4": "videos/a.mp4"})
    assert rows == [ac.VideoRow("videos/a.mp4", "MOM", "P01", "train")]
    # different names for the same things still work
    other = "signer,filename,word\nP09,b.mp4,DAD\n"
    assert ac.read_index(other, "test", {"b.mp4": "v/b.mp4"})[0].gloss == "DAD"
    # a listed video that is not in the archive is skipped, not a crash
    assert ac.read_index(csv_text, "train", {}) == []


def test_selection_filters_and_caps() -> None:
    rows = [ac.VideoRow(f"v{i}.mp4", g, f"P{i % 3}", "train")
            for g in ("mom", "dad", "nothere") for i in range(6)]
    kept, report = ac.select_rows(rows, labels=["MOM", "DAD", "AIRPLANE"], per_sign=2)
    assert len(kept) == 4                                   # 2 signs x 2 videos
    assert sorted(report.glosses_matched) == ["DAD", "MOM"]
    assert report.glosses_missing == ["AIRPLANE"]           # asked for, not in the data
    assert {r.gloss for r in kept} == {"mom", "dad"}        # "nothere" dropped


def test_selection_is_repeatable() -> None:
    rows = [ac.VideoRow(f"v{i}.mp4", "mom", f"P{i % 4}", "train") for i in range(20)]
    first, _ = ac.select_rows(rows, None, per_sign=5)
    second, _ = ac.select_rows(rows, None, per_sign=5)
    assert [r.member for r in first] == [r.member for r in second]


# ── end to end, from the zip ─────────────────────────────────────────────
def test_import_from_zip(tmp_path: Path, _no_mediapipe: _FakeTracker) -> None:
    archive = _mini_dataset(tmp_path)
    out = tmp_path / "out"
    report = ac.import_dataset(archive, out, labels=["MOM", "DAD", "DRINK", "AIRPLANE"],
                               window=16)

    assert report.converted == 12                      # 3 signs x 2 signers x 2 takes
    assert report.glosses_missing == ["AIRPLANE"]
    assert set(report.per_split) == {"train", "val", "test"}   # every split represented
    assert _no_mediapipe.closed                        # the tracker is released

    clips = np.load(next(out.glob("clips_*.npz")))["clips"]
    assert clips.shape[1:] == (16, RAW_DIM)
    assert clips.dtype == np.float16
    assert np.any(clips != 0)

    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 12
    assert {r["sign"] for r in rows} == {"MOM", "DAD", "DRINK"}
    assert {r["participant"] for r in rows} == set(PARTICIPANTS)

    saved = json.loads((out / "report.json").read_text())
    assert saved["clips_written"] == 12 and saved["missing_signs"] == ["AIRPLANE"]

    # every manifest row points at a real clip
    pairs = list(ac.iter_clips(out))
    assert len(pairs) == 12
    assert all(clip.shape == (16, RAW_DIM) for clip, _ in pairs)


def test_import_from_folder_matches_zip(tmp_path: Path) -> None:
    folder = _mini_dataset(tmp_path / "plain", as_zip=False)
    out = tmp_path / "out"
    report = ac.import_dataset(folder, out, labels=["MOM"], window=8)
    assert report.converted == 4
    assert np.load(next(out.glob("clips_*.npz")))["clips"].shape == (4, 8, RAW_DIM)


def test_resume_skips_finished_videos(tmp_path: Path) -> None:
    archive = _mini_dataset(tmp_path)
    out = tmp_path / "out"
    first = ac.import_dataset(archive, out, labels=["MOM"], window=8, limit=2)
    assert first.converted == 2
    second = ac.import_dataset(archive, out, labels=["MOM"], window=8)
    assert second.converted == 2                       # only the two not done before
    assert len(list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))) == 4


def test_clip_without_hands_is_dropped(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sign_bridge.capture.holistic.extract_features",
                        lambda _r: np.zeros(RAW_DIM, np.float32))
    archive = _mini_dataset(tmp_path)
    out = tmp_path / "out"
    report = ac.import_dataset(archive, out, labels=["MOM"], window=8)
    assert report.converted == 0 and report.skipped_no_hands == 4
    assert not list(out.glob("clips_*.npz"))


def test_unreadable_video_is_reported_not_fatal(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    archive = _mini_dataset(tmp_path)
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("corrupt frame")
        return np.full((6, RAW_DIM), 0.5, np.float32)

    monkeypatch.setattr(ac, "video_landmarks", flaky)
    report = ac.import_dataset(archive, tmp_path / "out", labels=["MOM"], window=8)
    assert report.failed == 1 and report.converted == 3


# ── tracking several videos at once ──────────────────────────────────────
def test_threads_match_one_at_a_time(tmp_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Four threads write exactly what one does: same clips, same order."""
    made: list[_FakeTracker] = []

    def factory() -> _FakeTracker:
        made.append(_FakeTracker(trim=False))
        return made[-1]

    monkeypatch.setattr(ac, "_holistic", factory)
    archive = _mini_dataset(tmp_path)
    one = ac.import_dataset(archive, tmp_path / "one", labels=["MOM", "DAD"],
                            window=8, workers=1)
    after_serial = len(made)
    many = ac.import_dataset(archive, tmp_path / "many", labels=["MOM", "DAD"],
                             window=8, workers=4)

    assert one.converted == 8
    assert many.as_dict() == one.as_dict()
    assert after_serial == 1                       # one at a time means one tracker
    assert len(made) - after_serial > 1            # the threaded run really used several
    assert all(tracker.closed for tracker in made)  # and released every one
    texts = [(tmp_path / side / "manifest.csv").read_text(encoding="utf-8")
             for side in ("one", "many")]
    assert texts[0] == texts[1]                    # same videos, same order, same shards
    clips = [np.load(next((tmp_path / side).glob("clips_*.npz")))["clips"]
             for side in ("one", "many")]
    assert np.array_equal(clips[0], clips[1])


def test_convert_many_reports_a_bad_video_without_stopping(tmp_path: Path) -> None:
    """A video that cannot even be read out of the zip is a message, not a crash."""
    archive = _mini_dataset(tmp_path)
    source = ac.Source(archive)
    try:
        real = next(iter(source.video_members().values()))
        results = list(ac.convert_many(source, ["nope.mp4", real], window=8,
                                       stride=1, workers=2))
        assert [name for name, _, _ in results] == ["nope.mp4", real]   # order kept
        assert results[0][1] is None and "KeyError" in results[0][2]
        assert results[1][1] is not None and results[1][2] == ""
    finally:
        source.close()


def test_default_workers_leaves_a_core_free() -> None:
    assert 1 <= ac.default_workers() <= max(1, (os.cpu_count() or 2) - 1)


def test_missing_split_files_is_a_clear_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(FileNotFoundError, match="No train/val/test CSV"):
        ac.import_dataset(empty, tmp_path / "out")


def test_holistic_error_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MediaPipe without Holistic must say exactly what to install."""
    import types

    monkeypatch.setattr(ac, "_holistic", REAL_HOLISTIC)    # undo the autouse stand-in
    fake = types.SimpleNamespace(__version__="1.2.3", solutions=types.SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "mediapipe", fake)
    with pytest.raises(RuntimeError, match="mediapipe==0.10.14"):
        ac._holistic()


def test_load_labels(tmp_path: Path) -> None:
    path = tmp_path / "asl.labels.json"
    path.write_text(json.dumps(["MOM", "DAD"]))
    assert ac.load_labels(path) == ["MOM", "DAD"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="list of signs"):
        ac.load_labels(bad)


# ── signs ASL Citizen names differently ──────────────────────────────────
def test_alias_map_skips_signs_that_are_wanted_themselves() -> None:
    m = ac.alias_map(["DAD", "MOM", "AWAKE"])
    assert m["FATHER"] == "DAD" and m["MOTHER"] == "MOM"
    assert "KITTEN" not in m                    # KITTY was not asked for
    # a candidate that is itself a wanted sign is never re-filed
    assert ac.alias_map(["WEUS", "WE"]).get("WE") is None


def test_select_rows_files_aliases_under_the_wanted_sign() -> None:
    rows = [ac.VideoRow(f"v{i}.mp4", g, "P1", "train")
            for i, g in enumerate(["FATHER", "FATHER", "MOM", "US", "WE"])]
    kept, report = ac.select_rows(rows, ["DAD", "MOM", "WEUS"], None,
                                  aliases=ac.alias_map(["DAD", "MOM", "WEUS"]))
    assert {(r.gloss, r.sign) for r in kept} == {("FATHER", "DAD"), ("MOM", "MOM"),
                                                 ("US", "WEUS"), ("WE", "WEUS")}
    assert report.glosses_missing == []
    assert report.aliased == {"DAD": ["FATHER"], "WEUS": ["US", "WE"]}
    # without aliases the same rows leave DAD and WEUS missing
    _, plain = ac.select_rows(rows, ["DAD", "MOM", "WEUS"], None)
    assert plain.glosses_missing == ["DAD", "WEUS"]


def test_import_with_aliases_writes_the_wanted_sign(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ac, "ALIAS_CANDIDATES", {"PAPA": ("DAD",)})
    archive = _mini_dataset(tmp_path)
    out = tmp_path / "out"
    report = ac.import_dataset(archive, out, labels=["PAPA", "MOM"], window=8, aliases=True)
    assert report.converted == 8 and report.glosses_missing == []
    rows = list(csv.DictReader((out / "manifest.csv").open(encoding="utf-8")))
    assert {(r["gloss"], r["sign"]) for r in rows} == {("DAD", "PAPA"), ("MOM", "MOM")}
    assert json.loads((out / "report.json").read_text())["aliased_signs"] == {"PAPA": ["DAD"]}


def test_resume_adds_aliases_to_an_existing_import(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """The real workflow: import direct matches first, then re-run with --aliases."""
    monkeypatch.setattr(ac, "ALIAS_CANDIDATES", {"PAPA": ("DAD",)})
    archive = _mini_dataset(tmp_path)
    out = tmp_path / "out"
    first = ac.import_dataset(archive, out, labels=["PAPA", "MOM"], window=8)
    assert first.converted == 4                      # MOM only
    second = ac.import_dataset(archive, out, labels=["PAPA", "MOM"], window=8, aliases=True)
    assert second.converted == 4                     # only the DAD videos are new
    assert len(list(ac.iter_clips(out))) == 8


# ── clips for the training notebook ──────────────────────────────────────
def _imported(tmp_path: Path) -> Path:
    """A small import folder: 2 shards, every split, a variant and an unknown sign."""
    out = tmp_path / "asl_citizen"
    out.mkdir()
    rows = []
    rng = np.random.default_rng(1)
    for shard in range(2):
        clips = rng.random((6, 8, RAW_DIM)).astype(np.float16)
        np.savez_compressed(out / f"clips_{shard:04d}.npz", clips=clips)
        for r in range(6):
            gloss = ["DAD", "MOM1", "MOM2", "HELLO", "CAT", "DAD"][r]
            rows.append({"shard": f"clips_{shard:04d}.npz", "row": str(r), "gloss": gloss,
                         "sign": {"MOM1": "MOM", "MOM2": "MOM"}.get(gloss, gloss),
                         "participant": f"P{shard}{r % 2}",
                         "split": ["train", "val", "test"][r % 3],
                         "source": f"{shard}_{r}.mp4"})
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "check.json").write_text(json.dumps({"keep_glosses": ["CAT", "DAD", "MOM1"],
                                                "drop_glosses": ["MOM2"]}), encoding="utf-8")
    return out


def test_training_clips_take_kept_glosses_of_the_chosen_splits(tmp_path: Path) -> None:
    out = _imported(tmp_path)
    labels = ["cat", "dad", "mom", "tv"]
    train = ac.training_clips(out, labels, ("train", "val"))
    # rows 0,1,3,4 of each shard are train/val; HELLO (3) is not a label
    assert sorted(train.glosses) == ["CAT", "CAT", "DAD", "DAD", "MOM1", "MOM1"]
    assert train.clips.shape == (6, 8, RAW_DIM) and train.clips.dtype == np.float16
    assert [labels[i].upper() for i in train.labels] == [g.rstrip("12") for g in train.glosses]
    test = ac.training_clips(out, labels, ("test",))
    assert test.glosses == ["DAD", "DAD"]                   # MOM2 was dropped by the check
    every = ac.training_clips(out, labels, ("test",), kept_only=False)
    assert sorted(every.glosses) == ["DAD", "DAD", "MOM2", "MOM2"]


def test_training_clips_match_the_stored_clips(tmp_path: Path) -> None:
    out = _imported(tmp_path)
    got = ac.training_clips(out, ["dad"], ("train",))
    first = np.load(out / "clips_0000.npz")["clips"][0]
    assert np.array_equal(got.clips[0], first) and got.participants[0] == "P00"


def test_training_clips_need_the_check(tmp_path: Path) -> None:
    out = _imported(tmp_path)
    (out / "check.json").unlink()
    with pytest.raises(FileNotFoundError, match="sign-check-aslcitizen"):
        ac.training_clips(out, ["dad"])
    assert len(ac.training_clips(out, ["tv"], kept_only=False)) == 0


def test_training_notebook_adds_asl_citizen() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "notebooks" / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("build_notebook", path)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    joined = "\n".join(builder.code_cells())
    for i, src in enumerate(builder.code_cells()):
        compile(src, f"<cell {i}>", "exec")
    assert 'training_clips(ASLC_DIR, LABELS, ("train", "val"))' in joined
    assert 'training_clips(ASLC_DIR, LABELS, ("test",))' in joined
    assert "make_ds(X_TRAIN, Y_TRAIN, train=True)" in joined      # training gets them
    assert "val_ds = make_ds(X[va], Y[va], train=False)" in joined  # validation does not


def test_run_c_notebook_changes_only_its_settings() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "notebooks" / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("build_notebook", path)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    base, run_c = builder.code_cells(), builder.code_cells("C")
    assert len(base) == len(run_c) and base[1:] == run_c[1:]      # same code, other settings
    for src in run_c:
        compile(src, "<run C>", "exec")
    assert '"SB_ARCHS", "conv_transformer"' in run_c[0] and '"SB_EPOCHS", 200' in run_c[0]
    assert '"SB_STRONG_AUG", "1"' in run_c[0] and '"SB_MOTION_LAGS", "1,2"' in run_c[0]
    assert '"SB_EPOCHS", 100' in base[0] and '"SB_STRONG_AUG", "0"' in base[0]


def test_run_d_notebook_is_run_c_with_another_training_seed() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "notebooks" / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("build_notebook", path)
    assert spec and spec.loader
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    run_c, run_d = builder.code_cells("C"), builder.code_cells("D")
    assert run_c[1:] == run_d[1:]                                   # same code
    assert run_c[0].replace('"SB_TRAIN_SEED", SEED', '"SB_TRAIN_SEED", 7') == run_d[0]
    for src in run_d:
        compile(src, "<run D>", "exec")
    joined = "\n".join(run_d)
    # the signer split and the live-check clips keep the fixed seed, so runs stay comparable
    assert "rng = np.random.default_rng(SEED)" in joined
    assert "live_idx = np.random.default_rng(SEED)" in joined
    assert "set_random_seed(TRAIN_SEED)" in joined and "seed=TRAIN_SEED" in joined
    assert "set_random_seed(SEED)" not in joined
    # earlier models are scored on the same test signers, alone and together
    assert "AveragedModel(members)" in joined and "ensemble_results.json" in joined
