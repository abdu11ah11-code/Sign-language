"""``sign-import`` — turn an external sign dataset into SignBridge training data.

Works with any dataset laid out as one folder per sign, holding photos
and/or videos. Built for **UAlpha40** (Urdu alphabet, Pakistan Sign
Language), whose layout is::

    UAlpha40/
        Alif-Original/   s0251-02alif.jpg ...      close-up hand photos
        Alif-Augmented/  s0251-02alif-Generated-1.jpg ...
        Jeem/            s0251.mp4 ...             letters that move: videos

What it does, per file:

* **Photo** → MediaPipe *Hands* (static-image mode; the photos are hand
  close-ups with no body, so Holistic would not find them) → one 21-point
  hand → a short "held" clip of ``landmarks.window_size`` frames with a
  little natural tremor added. A photo whose hand is cut off at the frame
  edge is retried with a padded border.
* **Video** → MediaPipe *Holistic* frame by frame → a motion clip, trimmed to
  the frames where a hand is visible (drops intro/outro cards).

Coordinates are converted so that one unit of x means the same distance as
in a live ``capture.frame_width × frame_height`` webcam frame. That keeps
hand shapes undistorted whether the source was a tall photo, a 16:9 video or
the 4:3 webcam.

The output goes to ``data/<language>/<LABEL>/import_<file>.csv`` — the same
format ``sign-collect`` writes — so ``sign-train`` uses it directly. Train
these with ``--features hand_shape`` (hand photos carry no body).

Re-running skips files that were already converted, so an interrupted import
resumes where it stopped.

Usage::

    sign-import --dataset "C:/Users/me/Downloads/UAlpha40" --language psl
    sign-import --dataset ... --language psl --max-per-class 40   # quicker
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sign_bridge.capture.holistic import FEATURE_DIM, RIGHT_HAND_SLICE
from sign_bridge.config import load_config
from sign_bridge.logging_conf import configure_logging, get_logger
from sign_bridge.typing import NDArray

log = get_logger(__name__)

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
_SUFFIX_RE = re.compile(r"[-_ ]+(original|augmented)$", re.IGNORECASE)
_MAX_IMAGE_SIDE = 960
_VIDEO_WIDTH = 960
_TREMOR = 0.01           # std of added jitter, as a fraction of hand size
_MIN_VIDEO_FRAMES = 8


# ── discovery ────────────────────────────────────────────────────────────
def label_from_folder(name: str) -> tuple[str, bool]:
    """Folder name → ``(LABEL, is_augmented)``.

    ``"Alif-Original"`` → ``("ALIF", False)``, ``"1-Hay-Augmented"`` →
    ``("1_HAY", True)``, ``"Jeem"`` → ``("JEEM", False)``.
    """
    match = _SUFFIX_RE.search(name.strip())
    augmented = bool(match and match.group(1).lower() == "augmented")
    base = name.strip()[: match.start()] if match else name.strip()
    label = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").upper()
    return label, augmented


def discover(
    root: Path, include_augmented: bool = False
) -> dict[str, list[Path]]:
    """Find every media file under ``root``, grouped by sign label.

    Any folder that directly contains photos or videos is a sign folder;
    ``-Original`` / ``-Augmented`` folders of the same sign are merged
    (augmented ones only when ``include_augmented``).
    """
    groups: dict[str, list[Path]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        media = sorted(
            Path(dirpath) / f for f in filenames
            if Path(f).suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
        )
        if not media:
            continue
        label, augmented = label_from_folder(Path(dirpath).name)
        if not label or (augmented and not include_augmented):
            continue
        groups.setdefault(label, []).extend(media)
    return {k: groups[k] for k in sorted(groups)}


def _spread(items: list[Path], limit: int | None) -> list[Path]:
    """Up to ``limit`` items spread evenly over the list (many signers)."""
    if not limit or len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


# ── coordinate handling ──────────────────────────────────────────────────
def to_frame_units(
    points: NDArray, width: int, height: int, aspect: float
) -> NDArray:
    """MediaPipe-normalized ``(n, 3)`` points → live-frame units.

    MediaPipe normalizes x (and z) by image *width* and y by *height*, so a
    raw hand from a tall photo is squashed compared with the same hand on a
    4:3 webcam. Scaling x and z by ``(width / height) / aspect`` puts every
    source in the units of a live ``aspect`` frame. Zero rows (undetected
    points) stay zero.
    """
    out = points.astype(np.float32).copy()
    factor = (width / height) / aspect
    out[:, 0] *= factor
    out[:, 2] *= factor
    return out


def _features_to_frame_units(
    feats: NDArray, width: int, height: int, aspect: float
) -> NDArray:
    return to_frame_units(feats.reshape(-1, 3), width, height, aspect).reshape(-1)


def static_clip(hand: NDArray, frames: int, rng: np.random.Generator) -> NDArray:
    """A held hand shape as a ``(frames, 162)`` clip with slight tremor.

    The hand goes in the right-hand slot; ``hand_shape`` features read
    whichever hand is visible.
    """
    base = hand.reshape(-1).astype(np.float32)
    size = float(np.max(np.linalg.norm(hand[:, :2] - hand[0, :2], axis=1))) or 1.0
    clip = np.zeros((frames, FEATURE_DIM), dtype=np.float32)
    clip[:, RIGHT_HAND_SLICE] = base + rng.normal(0.0, _TREMOR * size, (frames, base.size))
    return clip


# ── MediaPipe wrappers ───────────────────────────────────────────────────
def _mediapipe_solutions() -> Any:
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        raise RuntimeError(
            "This MediaPipe build has no 'solutions' API. "
            "Install the supported version: pip install mediapipe==0.10.14"
        )
    return solutions


def read_image(path: Path, max_side: int = _MAX_IMAGE_SIDE) -> NDArray | None:
    """Read a photo as RGB, honouring EXIF rotation, longest side ≤ max_side."""
    from PIL import Image, ImageOps

    try:
        with Image.open(path) as src:
            img = ImageOps.exif_transpose(src).convert("RGB")
            img.thumbnail((max_side, max_side))
            return np.asarray(img)
    except Exception as exc:  # noqa: BLE001 — corrupt/unsupported file
        log.warning("image_unreadable", path=str(path), error=str(exc))
        return None


class HandPhotoReader:
    """Detect the signing hand in close-up photos (MediaPipe Hands)."""

    def __init__(self, min_confidence: float = 0.3) -> None:
        hands = _mediapipe_solutions().hands
        self._hands = hands.Hands(
            static_image_mode=True, max_num_hands=2,
            model_complexity=1, min_detection_confidence=min_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def _detect(self, rgb: NDArray) -> NDArray | None:
        """Largest detected hand as ``(21, 3)`` normalized points, or None."""
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        best, best_area = None, -1.0
        for lms in result.multi_hand_landmarks:
            pts = np.array([(p.x, p.y, p.z) for p in lms.landmark], dtype=np.float32)
            area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
            if area > best_area:
                best, best_area = pts, area
        return best

    def hand(self, rgb: NDArray, aspect: float) -> NDArray | None:
        """Signing hand in live-frame units, or None if no hand is found.

        Close-ups often cut the hand at the edge, which defeats the palm
        detector; a second try adds a 30% border around the photo.
        """
        import cv2

        h, w = rgb.shape[:2]
        pts = self._detect(rgb)
        if pts is not None:
            return to_frame_units(pts, w, h, aspect)

        pad_y, pad_x = int(h * 0.3), int(w * 0.3)
        padded = cv2.copyMakeBorder(rgb, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        pts = self._detect(padded)
        if pts is None:
            return None
        ph, pw = padded.shape[:2]
        # Back to pixels of the original photo, then to normalized coords.
        px = pts.copy()
        px[:, 0] = (pts[:, 0] * pw - pad_x) / w
        px[:, 1] = (pts[:, 1] * ph - pad_y) / h
        px[:, 2] = pts[:, 2] * pw / w
        return to_frame_units(px, w, h, aspect)


def video_clip(path: Path, stride: int, aspect: float) -> NDArray | None:
    """Holistic features for a sign video, trimmed to frames with hands."""
    import cv2

    from sign_bridge.capture.holistic import extract_features

    holistic = _mediapipe_solutions().holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(path))
    rows: list[NDArray] = []
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % stride:
                continue
            h, w = frame.shape[:2]
            if w > _VIDEO_WIDTH:
                frame = cv2.resize(frame, (_VIDEO_WIDTH, int(h * _VIDEO_WIDTH / w)))
                h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feats = extract_features(holistic.process(rgb))
            rows.append(_features_to_frame_units(feats, w, h, aspect))
    finally:
        cap.release()
        holistic.close()
    if not rows:
        return None
    clip = np.stack(rows)
    visible = np.flatnonzero(np.any(clip[:, :126] != 0.0, axis=1))
    if visible.size < _MIN_VIDEO_FRAMES:
        return None
    clip = clip[visible[0] : visible[-1] + 1]
    if np.mean(np.any(clip[:, :126] != 0.0, axis=1)) < 0.5:
        return None
    return clip


# ── library seeding (Speech → Sign playback) ─────────────────────────────
def _still_to_video(rgb: NDArray, out: Path, seconds: float = 1.5, fps: int = 24) -> None:
    """Write a photo as a short letterboxed 640×480 clip."""
    import cv2

    canvas = np.full((480, 640, 3), (240, 245, 247), dtype=np.uint8)  # paper (BGR)
    h, w = rgb.shape[:2]
    scale = min(600 / w, 440 / h)
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (int(w * scale), int(h * scale)))
    y0, x0 = (480 - img.shape[0]) // 2, (640 - img.shape[1]) // 2
    canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))  # type: ignore[attr-defined]
    try:
        for _ in range(int(seconds * fps)):
            writer.write(canvas)
    finally:
        writer.release()


# ── main import loop ─────────────────────────────────────────────────────
@dataclass
class LabelReport:
    """Per-sign outcome of an import."""

    files: int = 0
    converted: int = 0
    skipped_existing: int = 0
    no_hand: int = 0
    unreadable: int = 0
    seconds: float = 0.0
    sources: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "sources"}
        d["sources"] = sorted(self.sources)
        return d


def import_dataset(
    dataset: Path,
    language: str,
    *,
    include_augmented: bool = False,
    max_per_class: int | None = None,
    video_stride: int = 2,
    seed_library: bool = True,
    overwrite: bool = False,
    config_path: str | None = None,
    only: Iterable[str] | None = None,
) -> dict[str, LabelReport]:
    """Convert ``dataset`` into ``data/<language>/<LABEL>/*.csv``.

    Returns a per-label report (also written next to the data as
    ``data/<language>_import_report.json``).
    """
    cfg = load_config(config_path)
    aspect = cfg.capture.frame_width / cfg.capture.frame_height
    frames = cfg.landmarks.window_size
    data_dir = cfg.resolve(cfg.paths.data_dir) / language
    lib_dir = cfg.resolve(cfg.paths.video_dir) / language
    rng = np.random.default_rng(cfg.training.seed)

    groups = discover(dataset, include_augmented)
    if only:
        wanted = {o.upper() for o in only}
        groups = {k: v for k, v in groups.items() if k in wanted}
    if not groups:
        raise SystemExit(f"No photos or videos found under {dataset}")

    total = sum(len(_spread(v, max_per_class)) for v in groups.values())
    print(f"Found {len(groups)} signs, {total} files to convert -> {data_dir}")

    reports: dict[str, LabelReport] = {}
    photo_reader: HandPhotoReader | None = None
    done = 0
    t_start = time.monotonic()
    try:
        for n, (label, files) in enumerate(groups.items(), start=1):
            rep = reports.setdefault(label, LabelReport())
            files = _spread(files, max_per_class)
            rep.files = len(files)
            out_dir = data_dir / label
            out_dir.mkdir(parents=True, exist_ok=True)
            library_clip = lib_dir / f"{label}.mp4"
            t0 = time.monotonic()

            for path in files:
                done += 1
                csv_path = out_dir / f"import_{re.sub(r'[^A-Za-z0-9_-]+', '_', path.stem)}.csv"
                if csv_path.exists() and not overwrite:
                    rep.skipped_existing += 1
                    continue
                ext = path.suffix.lower()
                clip: NDArray | None
                if ext in IMAGE_EXTS:
                    rep.sources.add("photos")
                    rgb = read_image(path)
                    if rgb is None:
                        rep.unreadable += 1
                        continue
                    if photo_reader is None:
                        photo_reader = HandPhotoReader()
                    hand = photo_reader.hand(rgb, aspect)
                    if hand is None:
                        rep.no_hand += 1
                        continue
                    clip = static_clip(hand, frames, rng)
                    if seed_library and not library_clip.exists():
                        _still_to_video(rgb, library_clip)
                else:
                    rep.sources.add("videos")
                    clip = video_clip(path, video_stride, aspect)
                    if clip is None:
                        rep.no_hand += 1
                        continue
                    if seed_library and not library_clip.exists():
                        library_clip.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, library_clip)
                np.savetxt(csv_path, clip, delimiter=",", fmt="%.6f")
                rep.converted += 1

            rep.seconds = time.monotonic() - t0
            elapsed = time.monotonic() - t_start
            eta = elapsed / max(done, 1) * (total - done)
            usable = rep.converted + rep.skipped_existing
            print(f"[{n:2d}/{len(groups)}] {label:<10} {usable:4d}/{rep.files:<4d} usable "
                  f"({'+'.join(sorted(rep.sources)) or '-'}; no hand {rep.no_hand}) "
                  f"{rep.seconds:5.1f}s  ETA {eta / 60:4.1f} min", flush=True)
    finally:
        if photo_reader is not None:
            photo_reader.close()

    report_path = data_dir.parent / f"{language}_import_report.json"
    report_path.write_text(
        json.dumps({k: v.as_dict() for k, v in reports.items()}, indent=2), encoding="utf-8"
    )
    usable = sum(r.converted + r.skipped_existing for r in reports.values())
    files_total = sum(r.files for r in reports.values())
    print(f"\nDone: {usable}/{files_total} files usable "
          f"({usable / max(files_total, 1):.0%}). Report: {report_path}")
    print(f"Next: sign-train --language {language} --features hand_shape")
    return reports


def main() -> None:
    """Console entry point for ``sign-import``."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Convert a folder-per-sign photo/video dataset (e.g. UAlpha40) "
                    "into SignBridge training data.")
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Folder containing one sub-folder per sign")
    parser.add_argument("--language", required=True, choices=["asl", "psl"])
    parser.add_argument("--include-augmented", action="store_true",
                        help="Also convert '-Augmented' folders (much slower; "
                             "sign-train already augments)")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Convert at most N files per sign (quicker test runs)")
    parser.add_argument("--video-stride", type=int, default=2,
                        help="Use every Nth video frame (default 2)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only these signs, e.g. --only ALIF BAY JEEM")
    parser.add_argument("--no-library", action="store_true",
                        help="Do not add clips to the Speech->Sign video library")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert files that were already imported")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if not args.dataset.is_dir():
        raise SystemExit(f"Not a folder: {args.dataset}")
    import_dataset(
        args.dataset, args.language,
        include_augmented=args.include_augmented,
        max_per_class=args.max_per_class,
        video_stride=max(1, args.video_stride),
        seed_library=not args.no_library,
        overwrite=args.overwrite,
        config_path=args.config,
        only=args.only,
    )


if __name__ == "__main__":
    main()
