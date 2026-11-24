"""``sign-import-aslcitizen`` — ASL Citizen videos → SignBridge landmark clips.

`ASL Citizen <https://www.microsoft.com/en-us/research/project/asl-citizen/>`_
is 83,399 **webcam** videos of 2,731 signs by 52 Deaf signers. The Google ASL
Signs data SignBridge trains on was filmed on phones, so ASL Citizen covers
exactly the gap the live app runs into: ordinary laptop webcams.

Why this runs locally and not on Kaggle:

* the download is a single **42.8 GB** zip — more than a Kaggle session's disk,
  and Microsoft's download page gives no stable direct link to fetch from;
* the landmarks must come from the **same** MediaPipe Holistic pipeline the app
  uses, and ``mediapipe.solutions.holistic`` was removed after 0.10.21, which
  in turn needs Python ≤ 3.12 — Kaggle now runs 3.13. SignBridge pins 0.10.14,
  so the machine that runs the app is already the right machine.

**The zip is never unpacked.** Each wanted video is read straight out of it,
written to a temporary file for OpenCV, converted, and deleted — so this needs
only the 42.8 GB of the zip itself, not another 42.8 GB beside it.

Tracking costs about 30 ms a frame, so ``--workers`` (several videos at once,
see :func:`convert_many`) is what decides whether a run takes hours or most of
a day. A run can be stopped with Ctrl-C and started again: anything already in
``manifest.csv`` is skipped.

Output: ``<out>/clips_XXXX.npz`` shards holding ``(N, window, 282)`` float16
clips in SignBridge's raw landmark layout, plus ``manifest.csv`` (one row per
video: gloss, participant, split, source file) and ``report.json``. Upload that
folder to Kaggle as a dataset and train on it beside ASL Signs.

**Licence.** ASL Citizen is released for **non-commercial research only**, and
may not be redistributed. Keep both the videos and anything derived from them
private; publish results, not data.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import zipfile
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from sign_bridge.logging_conf import configure_logging, get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)

#: Frames per saved clip (matches the ASL Signs models' window).
DEFAULT_WINDOW = 64
#: Clips per output shard — small enough to write progress often.
SHARD_SIZE = 512
#: Longest side a frame is scaled to before tracking (speed vs. accuracy).
VIDEO_WIDTH = 640
#: A clip is dropped unless a hand is visible in at least this share of frames.
MIN_HAND_COVERAGE = 0.5
#: ...and unless it has at least this many frames with a hand.
MIN_HAND_FRAMES = 4

_VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".avi", ".mkv")


# ── reading the dataset's own index ──────────────────────────────────────
def normalize_gloss(text: str) -> str:
    """``"ice cream"``, ``"ICE_CREAM"``, ``"ice-cream1"`` → ``"ICECREAM"``.

    ASL Citizen writes glosses as ASL-LEX entries (sometimes with a
    disambiguating digit); ASL Signs writes them as plain words. Comparing
    them stripped of case, spacing and punctuation matches far more of them.
    """
    return re.sub(r"[^A-Z0-9]", "", text.strip().upper()).rstrip("0123456789")


def _pick_column(header: list[str], *wanted: str) -> str | None:
    """Find the column meaning one of ``wanted``, exact names winning.

    Exact matches are tried before partial ones, because a partial match alone
    picks the wrong column: a file listing a ``signer`` column would otherwise
    answer to "sign" and be read as the gloss.
    """
    clean = {name: re.sub(r"[^a-z0-9]", "", name.strip().lower()) for name in header}
    for want in wanted:
        for name, simple in clean.items():
            if simple == want:
                return name
    for want in wanted:
        for name, simple in clean.items():
            if want in simple:
                return name
    return None


@dataclass(frozen=True)
class VideoRow:
    """One video listed in an ASL Citizen split file."""

    member: str      # path inside the zip, or relative to the folder
    gloss: str       # as written in the CSV
    participant: str
    split: str       # train / val / test, from the file it was listed in
    sign: str = ""   # the SignBridge sign it is filed under (set by select_rows)


#: ASL Citizen glosses that may be the same sign as a SignBridge label under
#: another name (ASL Citizen follows ASL-LEX, Google's list a children's
#: vocabulary). These are **candidates only**: an English synonym is not
#: always the same sign, so ``sign-check-aslcitizen`` tests each one against
#: the trained model before any of it is used for training. A candidate that
#: is itself one of the wanted signs is never aliased (AWAKE stays AWAKE).
ALIAS_CANDIDATES: dict[str, tuple[str, ...]] = {
    "DAD": ("FATHER",), "MOM": ("MOTHER",),
    "GRANDMA": ("GRANDMOTHER",), "GRANDPA": ("GRANDFATHER",),
    "ALLIGATOR": ("CROCODILE",), "ANOTHER": ("OTHER",),
    "EYE": ("EYES",), "SHOE": ("SHOES",), "LOOK": ("LOOKAT",),
    "WEUS": ("WE", "US"), "MINEMY": ("MY", "MINE"), "HESHEIT": ("HE",),
    "HAVETO": ("MUST",), "GLASSWINDOW": ("WINDOW",), "TOOTH": ("TEETH",),
    "POTTY": ("TOILET",), "BESIDE": ("NEXTTO",), "CALLONPHONE": ("PHONE",),
    "OWIE": ("HURT",), "NOISY": ("NOISE",), "GARBAGE": ("TRASH",),
    "POLICE": ("POLICEMAN",), "NUTS": ("NUT",), "PUZZLE": ("JIGSAWPUZZLE",),
    "KITTY": ("KITTEN",),
}


def alias_map(labels: list[str] | None) -> dict[str, str]:
    """Normalized ASL Citizen gloss → the wanted sign it is a candidate for."""
    wanted = {normalize_gloss(lbl) for lbl in labels} if labels else None
    out: dict[str, str] = {}
    for sign, glosses in ALIAS_CANDIDATES.items():
        if wanted is not None and sign not in wanted:
            continue
        for gloss in glosses:
            key = normalize_gloss(gloss)
            if wanted is None or key not in wanted:
                out[key] = sign
    return out


def read_index(text: str, split: str, video_members: dict[str, str]) -> list[VideoRow]:
    """Parse one split CSV into :class:`VideoRow` entries.

    The column names in the release are not documented, so they are detected
    by what their names contain rather than assumed.

    Args:
        text: The CSV's contents.
        split: ``"train"``, ``"val"`` or ``"test"``.
        video_members: Lower-case file name → path inside the zip, used to
            find each listed video wherever it actually sits.

    Returns:
        The rows whose video could be located.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return []
    header = list(rows[0])
    video_col = _pick_column(header, "videofile", "video", "filename", "file", "name")
    gloss_col = _pick_column(header, "gloss", "word", "label", "sign")
    part_col = _pick_column(header, "participantid", "participant", "signer", "user", "subject")
    if video_col is None or gloss_col is None:
        log.warning("asl_citizen_columns_not_found", split=split, header=header)
        return []
    out = []
    for row in rows:
        name = str(row.get(video_col) or "").strip()
        member = video_members.get(Path(name).name.lower())
        if not name or member is None:
            continue
        out.append(VideoRow(member=member, gloss=str(row.get(gloss_col) or "").strip(),
                            participant=str(row.get(part_col) or "").strip() if part_col else "",
                            split=split))
    return out


def _split_name(filename: str) -> str | None:
    low = filename.lower()
    for split in ("train", "val", "test"):
        if split in low:
            return "val" if split == "val" else split
    return None


class Source:
    """The dataset, whether it is still a zip or an unpacked folder.

    Safe to share between threads: one zip holds a single file handle that
    seeks as it reads, so every read is taken under a lock. Pulling one video
    out takes a few milliseconds against seconds of tracking, so the threads
    barely queue for it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._zip = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None
        if self._zip is not None:
            names = self._zip.namelist()
        else:
            names = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]
        self._names = names

    def close(self) -> None:
        """Release the zip handle."""
        if self._zip is not None:
            self._zip.close()

    def video_members(self) -> dict[str, str]:
        """Lower-case file name → where that video lives in the source."""
        return {Path(n).name.lower(): n for n in self._names
                if Path(n).suffix.lower() in _VIDEO_SUFFIXES}

    def split_files(self) -> dict[str, str]:
        """Split name → the CSV listing it."""
        found: dict[str, str] = {}
        for name in self._names:
            if not name.lower().endswith(".csv"):
                continue
            split = _split_name(Path(name).name)
            if split and split not in found:
                found[split] = name
        return found

    def read_text(self, member: str) -> str:
        """Read a text member (CSV)."""
        if self._zip is None:
            return (self.path / member).read_text(encoding="utf-8-sig", errors="replace")
        with self._lock:
            data = self._zip.read(member)
        return data.decode("utf-8-sig", errors="replace")

    def open_video(self, member: str) -> Path:
        """Put a video somewhere OpenCV can open it; delete it when done."""
        if self._zip is None:
            return self.path / member
        with self._lock:
            data = self._zip.read(member)
        suffix = Path(member).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(data)
            return Path(handle.name)

    def release_video(self, path: Path) -> None:
        """Delete a temporary copy made by :meth:`open_video`."""
        if self._zip is not None:
            path.unlink(missing_ok=True)


# ── video → landmarks ────────────────────────────────────────────────────
def _holistic() -> Any:
    """A MediaPipe Holistic tracker, or a clear error if it is unavailable."""
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "holistic"):
        raise RuntimeError(
            f"mediapipe {getattr(mp, '__version__', '?')} has no 'solutions.holistic'. "
            "It was removed after 0.10.21 (which needs Python <= 3.12). "
            "Install the pinned version with: pip install mediapipe==0.10.14")
    return solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def video_landmarks(path: Path, tracker: Any, stride: int = 1) -> NDArray | None:
    """Track one video into ``(frames, 282)`` raw landmarks, or None if unusable.

    The clip is trimmed to the stretch where a hand is actually visible, so
    the still moments before and after the sign are dropped.
    """
    import cv2

    from sign_bridge.capture.holistic import extract_features

    cap = cv2.VideoCapture(str(path))
    rows: list[NDArray] = []
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            if (index - 1) % stride:
                continue
            height, width = frame.shape[:2]
            if width > VIDEO_WIDTH:
                frame = cv2.resize(frame, (VIDEO_WIDTH, int(height * VIDEO_WIDTH / width)))
            rows.append(extract_features(tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    finally:
        cap.release()
    if not rows:
        return None
    clip = np.stack(rows)
    hands = np.any(clip[:, :126] != 0.0, axis=1)
    visible = np.flatnonzero(hands)
    if visible.size < MIN_HAND_FRAMES:
        return None
    clip = clip[visible[0] : visible[-1] + 1]
    if float(np.mean(np.any(clip[:, :126] != 0.0, axis=1))) < MIN_HAND_COVERAGE:
        return None
    return clip


# ── the import itself ────────────────────────────────────────────────────
@dataclass
class ImportReport:
    """What happened, for the console and ``report.json``."""

    listed: int = 0
    wanted: int = 0
    converted: int = 0
    skipped_no_hands: int = 0
    failed: int = 0
    glosses_matched: list[str] = field(default_factory=list)
    glosses_missing: list[str] = field(default_factory=list)
    per_split: dict[str, int] = field(default_factory=dict)
    aliased: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {"videos_listed": self.listed, "videos_wanted": self.wanted,
                "clips_written": self.converted, "skipped_no_hands": self.skipped_no_hands,
                "failed_to_read": self.failed, "per_split": self.per_split,
                "signs_found": len(self.glosses_matched),
                "signs_missing": len(self.glosses_missing),
                "missing_signs": sorted(self.glosses_missing),
                "aliased_signs": {k: sorted(v) for k, v in sorted(self.aliased.items())}}


def select_rows(rows: list[VideoRow], labels: list[str] | None,
                per_sign: int | None, seed: int = 42,
                aliases: dict[str, str] | None = None) -> tuple[list[VideoRow], ImportReport]:
    """Keep the videos we want: the wanted signs, capped per sign.

    Args:
        rows: Everything the split files list.
        labels: Signs to keep (SignBridge glosses); None keeps all.
        per_sign: Most videos per sign, spread across signers; None keeps all.
        seed: Makes the choice repeatable.
        aliases: Normalized ASL Citizen gloss → wanted sign, for signs named
            differently (see :func:`alias_map`). Each kept row records the
            sign it was filed under in ``sign``.
    """
    import dataclasses

    report = ImportReport(listed=len(rows))
    wanted = {normalize_gloss(lbl) for lbl in labels} if labels else None
    aliases = aliases or {}
    by_sign: dict[str, list[VideoRow]] = {}
    for row in rows:
        key = normalize_gloss(row.gloss)
        if (wanted is None or key not in wanted) and key in aliases:
            report.aliased.setdefault(aliases[key], [])
            if row.gloss not in report.aliased[aliases[key]]:
                report.aliased[aliases[key]].append(row.gloss)
            key = aliases[key]
        if wanted is not None and key not in wanted:
            continue
        by_sign.setdefault(key, []).append(dataclasses.replace(row, sign=key))

    rng = np.random.default_rng(seed)
    kept: list[VideoRow] = []
    for key, group in sorted(by_sign.items()):
        if per_sign is not None and len(group) > per_sign:
            # spread the choice over signers so one person cannot dominate a sign
            group = sorted(group, key=lambda r: (r.participant, r.member))
            picked = rng.choice(len(group), size=per_sign, replace=False)
            group = [group[i] for i in sorted(picked)]
        kept.extend(group)
        report.glosses_matched.append(key)
    if wanted is not None:
        report.glosses_missing = sorted(wanted - set(by_sign))
    report.wanted = len(kept)
    return kept, report


def _write_shard(out: Path, index: int, clips: list[NDArray]) -> None:
    np.savez_compressed(out / f"clips_{index:04d}.npz", clips=np.stack(clips))


# ── one video, and how to do many at once ────────────────────────────────
#: Tracked result for one video: ``(source member, clip or None, error text)``.
Converted = tuple[str, NDArray | None, str]


def default_workers() -> int:
    """Threads to track with by default: every core but one, at least one.

    Tracking is the whole cost of this job (~30 ms a frame, one core's worth),
    so a single-threaded import of 7,500 videos runs for hours. One core is
    left over so the machine stays usable while it does.
    """
    return max(1, (os.cpu_count() or 2) - 1)


def _convert(source: Source, member: str, tracker: Any,
             window: int, stride: int) -> tuple[NDArray | None, str]:
    """Track one video into a float16 ``(window, 282)`` clip.

    Returns:
        ``(clip, "")`` on success, ``(None, "")`` when no hand was visible,
        or ``(None, error)`` when the video could not be read — a single bad
        file in 83,399 must not end a run that takes hours.
    """
    from sign_bridge.data.preprocess import resample_clip

    temp: Path | None = None
    try:
        # Getting the video out of the zip can fail too (a corrupt entry, a full
        # temp disk), so it is inside the guard, not before it.
        temp = source.open_video(member)
        clip = video_landmarks(temp, tracker, stride)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if temp is not None:
            source.release_video(temp)
    if clip is None:
        return None, ""
    return resample_clip(clip, window).astype(np.float16), ""


def convert_many(source: Source, members: list[str], window: int, stride: int,
                 workers: int = 1) -> Iterator[Converted]:
    """Track the videos, up to ``workers`` at a time, yielding them in order.

    **Threads, not processes.** MediaPipe releases the GIL while its C++ graph
    runs — measured on this pipeline, two threads on two cores tracked 1.9x as
    many frames a second as one — so threads buy the same speedup as worker
    processes. They also avoid what makes ``multiprocessing`` brittle here:
    with the ``spawn`` start method (the only one Windows has) every worker
    re-imports the program, which hangs a script without an
    ``if __name__ == "__main__"`` guard and cannot work from a notebook at all.

    Each thread keeps its own tracker, because a Holistic instance carries the
    tracking state of one video and cannot be shared. At most ``workers * 2``
    finished clips are held in memory at once, so the order the results come
    back in never costs more than that.
    """
    if workers <= 1:
        tracker = _holistic()
        try:
            for member in members:
                clip, error = _convert(source, member, tracker, window, stride)
                yield member, clip, error
        finally:
            tracker.close()
        return

    local = threading.local()
    trackers: list[Any] = []
    lock = threading.Lock()

    def convert(member: str) -> Converted:
        tracker = getattr(local, "tracker", None)
        if tracker is None:
            tracker = local.tracker = _holistic()
            with lock:
                trackers.append(tracker)      # so they can all be closed at the end
        clip, error = _convert(source, member, tracker, window, stride)
        return member, clip, error

    queue = iter(members)
    try:
        with ThreadPoolExecutor(workers, thread_name_prefix="track") as pool:
            waiting: deque[Future[Converted]] = deque(
                pool.submit(convert, m) for m in islice(queue, workers * 2))
            while waiting:
                result = waiting.popleft().result()
                for member in islice(queue, 1):
                    waiting.append(pool.submit(convert, member))
                yield result
    finally:
        for tracker in trackers:
            tracker.close()


def import_dataset(
    source_path: Path, out: Path, labels: list[str] | None = None,
    window: int = DEFAULT_WINDOW, per_sign: int | None = None, stride: int = 1,
    limit: int | None = None, resume: bool = True,
    progress: Any = None, workers: int = 1, aliases: bool = False,
) -> ImportReport:
    """Convert ASL Citizen videos into landmark clips.

    Args:
        source_path: ``ASL_Citizen.zip`` or the folder it unpacks to.
        out: Where the shards, manifest and report are written.
        labels: Signs to keep (e.g. the 250 ASL Signs glosses); None keeps all.
        window: Frames each clip is resampled to.
        per_sign: Most videos per sign.
        stride: Track every Nth frame (2 halves the time, coarser motion).
        limit: Stop after this many videos (for a quick trial).
        resume: Continue a previous run instead of redoing it.
        progress: Called as ``progress(done, total, report)`` now and then.
        workers: Videos to track at the same time, each on its own thread
            (see :func:`default_workers` and :func:`convert_many`).
        aliases: Also import the :data:`ALIAS_CANDIDATES` — ASL Citizen signs
            that may be a wanted sign under another name. Check them with
            ``sign-check-aslcitizen`` before training on them.

    Returns:
        An :class:`ImportReport`.
    """
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    done: set[str] = set()
    if resume and manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            done = {row["source"] for row in csv.DictReader(handle)}
        log.info("asl_citizen_resume", already_done=len(done))

    source = Source(source_path)
    try:
        members = source.video_members()
        splits = source.split_files()
        if not splits:
            raise FileNotFoundError(
                f"No train/val/test CSV found in {source_path}. Expected the split files "
                "that ship with ASL Citizen.")
        rows: list[VideoRow] = []
        for split, member in sorted(splits.items()):
            rows.extend(read_index(source.read_text(member), split, members))
        kept, report = select_rows(rows, labels, per_sign,
                                   aliases=alias_map(labels) if aliases else None)
        if limit is not None:
            kept = kept[:limit]
            report.wanted = len(kept)

        todo = [row for row in kept if row.member not in done]
        by_member = {row.member: row for row in todo}
        todo_members = [row.member for row in todo]
        workers = max(1, min(workers, len(todo_members))) if todo_members else 1
        log.info("asl_citizen_start", videos=len(todo_members), workers=workers)

        shard_index = len(list(out.glob("clips_*.npz")))
        buffer: list[NDArray] = []
        new_rows: list[dict[str, str]] = []
        results = convert_many(source, todo_members, window, stride, workers)
        for position, (member, clip, error) in enumerate(results, 1):
            row = by_member[member]
            if error:
                log.warning("asl_citizen_video_failed", video=member, error=error)
                report.failed += 1
            elif clip is None:
                report.skipped_no_hands += 1
            else:
                buffer.append(clip)
                new_rows.append({"shard": f"clips_{shard_index:04d}.npz",
                                 "row": str(len(buffer) - 1), "gloss": row.gloss,
                                 "sign": row.sign or normalize_gloss(row.gloss),
                                 "participant": row.participant, "split": row.split,
                                 "source": row.member})
                report.converted += 1
                report.per_split[row.split] = report.per_split.get(row.split, 0) + 1
            if len(buffer) >= SHARD_SIZE:
                _write_shard(out, shard_index, buffer)
                _append_manifest(manifest_path, new_rows)
                buffer, new_rows, shard_index = [], [], shard_index + 1
            if progress is not None and position % 50 == 0:
                progress(position, len(todo_members), report)
        if buffer:
            _write_shard(out, shard_index, buffer)
            _append_manifest(manifest_path, new_rows)
    finally:
        source.close()

    (out / "report.json").write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return report


def _append_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def load_labels(path: Path) -> list[str]:
    """Read a model's ``*.labels.json`` (or any JSON list of signs)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not hold a list of signs")
    return [str(x) for x in data]


def iter_clips(out: Path) -> Iterator[tuple[NDArray, dict[str, str]]]:
    """Walk an imported folder, yielding ``(clip, manifest row)``."""
    manifest = out / "manifest.csv"
    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cache: dict[str, NDArray] = {}
    for row in rows:
        shard = row["shard"]
        if shard not in cache:
            cache = {shard: np.load(out / shard)["clips"]}
        yield cache[shard][int(row["row"])], row


@dataclass
class TrainingClips:
    """Imported ASL Citizen clips ready for training or testing.

    Attributes:
        clips: ``(N, window, 282)`` raw landmark clips (float16).
        labels: Class index of each clip in the model's label list.
        participants: ASL Citizen participant of each clip.
        glosses: ASL Citizen gloss of each clip (e.g. ``ARM1``).
    """

    clips: NDArray
    labels: NDArray
    participants: list[str]
    glosses: list[str]

    def __len__(self) -> int:
        return len(self.labels)


def training_clips(folder: Path, labels: list[str], splits: tuple[str, ...] = ("train", "val"),
                   kept_only: bool = True) -> TrainingClips:
    """The imported clips of the given ASL Citizen splits, labelled for a model.

    Args:
        folder: An import folder (``manifest.csv`` and the shards).
        labels: The model's label list; clips of other signs are left out.
        splits: ASL Citizen splits to take. Its splits keep signers apart, so
            ``("train", "val")`` adds new people to training and ``("test",)``
            is a test set of people seen nowhere else.
        kept_only: Only glosses that ``sign-check-aslcitizen`` kept
            (``check.json``); its decision used the train and val splits only,
            so filtering the test split by it does not peek at test results.

    Raises:
        FileNotFoundError: ``kept_only`` without a ``check.json`` in ``folder``.
    """
    keep: set[str] | None = None
    if kept_only:
        check = folder / "check.json"
        if not check.is_file():
            raise FileNotFoundError(f"{check} not found: run sign-check-aslcitizen first")
        keep = set(json.loads(check.read_text(encoding="utf-8"))["keep_glosses"])
    index = {label.upper(): i for i, label in enumerate(labels)}
    clips: list[NDArray] = []
    ys: list[int] = []
    people: list[str] = []
    glosses: list[str] = []
    for clip, row in iter_clips(folder):
        sign = row["sign"].upper()
        if row["split"] not in splits or sign not in index:
            continue
        if keep is not None and row["gloss"] not in keep:
            continue
        clips.append(clip)
        ys.append(index[sign])
        people.append(row["participant"])
        glosses.append(row["gloss"])
    window = clips[0].shape[0] if clips else DEFAULT_WINDOW
    return TrainingClips(
        clips=np.stack(clips) if clips else np.zeros((0, window, 282), np.float16),
        labels=np.asarray(ys, dtype=np.int64), participants=people, glosses=glosses)


# ── CLI ──────────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for ``sign-import-aslcitizen``."""
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="sign-import-aslcitizen",
        description="Convert ASL Citizen videos into SignBridge landmark clips. "
                    "Reads straight out of ASL_Citizen.zip — no need to unpack it.",
        epilog="ASL Citizen is licensed for non-commercial research only and may not be "
               "redistributed. Keep the output private; publish results, not data.")
    parser.add_argument("--source", type=Path, required=True,
                        help="ASL_Citizen.zip, or the folder it unpacks to")
    parser.add_argument("--out", type=Path, default=Path("data/asl_citizen"),
                        help="where to write the clips (default: data/asl_citizen)")
    parser.add_argument("--labels", type=Path,
                        help="JSON list of signs to keep, e.g. a model's *.labels.json "
                             "(default: keep every sign)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"frames per clip (default: {DEFAULT_WINDOW})")
    parser.add_argument("--per-sign", type=int,
                        help="most videos per sign (default: all)")
    parser.add_argument("--stride", type=int, default=1,
                        help="track every Nth frame; 2 is about twice as fast (default: 1)")
    parser.add_argument("--limit", type=int,
                        help="stop after this many videos — use it for a quick trial first")
    parser.add_argument("--restart", action="store_true",
                        help="ignore an existing manifest and start over")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help=f"videos to track side by side (default on this machine: "
                             f"{default_workers()})")
    parser.add_argument("--aliases", action="store_true",
                        help="also import signs ASL Citizen names differently (FATHER for "
                             "DAD, ...). Candidates only: check them with "
                             "sign-check-aslcitizen before training")
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else None

    started = time.monotonic()

    def show(done: int, total: int, report: ImportReport) -> None:
        elapsed = time.monotonic() - started
        left = (total - done) * elapsed / done if done else 0.0
        print(f"  {done:,}/{total:,} videos — {report.converted:,} clips, "
              f"{report.skipped_no_hands:,} without hands, {report.failed:,} unreadable"
              f" — about {left / 60:,.0f} min left", flush=True)

    report = import_dataset(
        args.source, args.out, labels=labels, window=args.window, per_sign=args.per_sign,
        stride=args.stride, limit=args.limit, resume=not args.restart, progress=show,
        workers=args.workers, aliases=args.aliases)

    print(f"\n{report.converted:,} clips written to {args.out}")
    if report.aliased:
        print("  imported under another name (check before training):")
        for sign, glosses in sorted(report.aliased.items()):
            print(f"    {sign:12s} <- {', '.join(sorted(glosses))}")
    print(f"  took {(time.monotonic() - started) / 60:,.0f} min "
          f"on {args.workers} worker{'s' if args.workers != 1 else ''}")
    print(f"  signs found: {len(report.glosses_matched)}"
          + (f" of {len(report.glosses_matched) + len(report.glosses_missing)} asked for"
             if labels else ""))
    print(f"  per split: {report.per_split}")
    if report.skipped_no_hands or report.failed:
        print(f"  dropped: {report.skipped_no_hands:,} with no hand visible, "
              f"{report.failed:,} unreadable")
    if report.glosses_missing:
        shown = sorted(report.glosses_missing)[:12]
        print(f"  not in ASL Citizen ({len(report.glosses_missing)}): {', '.join(shown)}"
              + (" ..." if len(report.glosses_missing) > 12 else ""))
    print("\nASL Citizen is for non-commercial research only — keep this folder private.")


if __name__ == "__main__":  # pragma: no cover
    main()
