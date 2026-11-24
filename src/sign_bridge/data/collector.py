"""``sign-collect`` — capture training data and sign videos in one pass.

Usage:
    sign-collect --label HELLO --mode asl --samples 50

For each repetition the tool shows a live preview with a 3-2-1 countdown,
records a fixed-length clip, and writes **two** artifacts:

    data/<lang>/<LABEL>/<timestamp>_<rep>.csv   landmark features (training)
    recordings/<lang>/<LABEL>/<...>.mp4         raw video (audit/debug)

Each clip covers ``collector.clip_seconds`` of *wall-clock* time (however
many frames the machine manages), which is exactly the window the live
recognizer resamples — so training and live input move at the same speed.

Quality gate: a take where hands were tracked in fewer than
``collector.min_hands_visible`` of frames is discarded and re-recorded,
because a clip without hands teaches the model nothing.

The first good repetition (hands visible in ≥80% of frames) is also copied
to ``assets/sign_videos/<lang>/<LABEL>.mp4`` so collecting data
simultaneously builds the text→sign playback library.

Keys during the countdown:  q / ESC — abort    s — skip this rep
"""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from sign_bridge.capture.holistic import HolisticTracker
from sign_bridge.capture.webcam import CameraError, ThreadedCamera
from sign_bridge.config import load_config
from sign_bridge.logging_conf import configure_logging, get_logger
from sign_bridge.typing import NDArray
from sign_bridge.ui.overlay import draw_landmarks

log = get_logger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INK = (17, 22, 24)        # BGR of Field Guide ink #181611
_PAPER = (240, 245, 247)   # BGR of Field Guide paper #F7F5F0
_GREEN = (69, 90, 30)      # BGR of Field Guide green #1E5A45


def _banner(frame: NDArray, text: str, sub: str = "") -> NDArray:
    """Draw a paper-colored banner with ink text at the top of the frame."""
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 64), _PAPER, thickness=-1)
    cv2.putText(out, text, (16, 30), _FONT, 0.8, _INK, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (16, 54), _FONT, 0.5, _GREEN, 1, cv2.LINE_AA)
    return out


class _Countdown(Enum):
    GO = "go"
    SKIP = "skip"
    ABORT = "abort"


def _countdown(
    camera: ThreadedCamera, seconds: int, label: str, rep: int, total: int, note: str = ""
) -> _Countdown:
    """Show a countdown on the preview; the user may skip or abort."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        remaining = int(end - time.monotonic()) + 1
        frame = _banner(
            camera.read(),
            f"{label}  -  rep {rep}/{total}",
            note or f"recording in {remaining}...  (q to quit, s to skip)",
        )
        cv2.putText(
            frame, str(remaining),
            (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
            _FONT, 4.0, _GREEN, 8, cv2.LINE_AA,
        )
        cv2.imshow("sign-collect", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return _Countdown.ABORT
        if key == ord("s"):
            return _Countdown.SKIP
    return _Countdown.GO


def _record_clip(
    camera: ThreadedCamera,
    tracker: HolisticTracker,
    seconds: float,
    video_path: Path,
    fps: int,
    label: str,
) -> tuple[NDArray, float]:
    """Record ``seconds`` of wall-clock motion.

    Returns:
        ``(features (T, 162), hands_visible_ratio)``. ``T`` depends on how
        fast this machine runs MediaPipe; training resamples every clip to
        ``landmarks.window_size`` frames anyway.
    """
    video_path.parent.mkdir(parents=True, exist_ok=True)
    first = camera.read()
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps,
        (first.shape[1], first.shape[0]),
    )
    rows: list[NDArray] = []
    visible = 0
    interval = 1.0 / fps
    start = time.monotonic()
    next_t = start
    try:
        while (elapsed := time.monotonic() - start) < seconds:
            while time.monotonic() < next_t:
                time.sleep(0.001)
            next_t = max(next_t + interval, time.monotonic())

            frame = camera.read()
            result = tracker.process(frame)
            rows.append(result.features)
            visible += int(result.hands_visible)
            writer.write(frame)

            preview = draw_landmarks(frame, result.raw)
            preview = _banner(preview, f"REC  {label}", f"{elapsed:.1f} / {seconds:.1f} s")
            cv2.circle(preview, (preview.shape[1] - 30, 32), 10, (47, 53, 255), -1)
            cv2.imshow("sign-collect", preview)
            cv2.waitKey(1)
    finally:
        writer.release()
    feats = np.stack(rows)
    return feats, visible / max(1, len(rows))


def collect(label: str, language: str, samples: int, config_path: str | None = None) -> int:
    """Run the interactive collection session. Returns reps recorded."""
    cfg = load_config(config_path)
    label = label.strip().upper().replace(" ", "_")
    fps = cfg.capture.target_fps

    data_dir = cfg.resolve(cfg.paths.data_dir) / language / label
    rec_dir = cfg.resolve(cfg.paths.recordings_dir) / language / label
    library_clip = cfg.resolve(cfg.paths.video_dir) / language / f"{label}.mp4"
    data_dir.mkdir(parents=True, exist_ok=True)

    recorded = 0
    rejected = 0
    try:
        with ThreadedCamera(
            index=cfg.capture.camera_index,
            width=cfg.capture.frame_width,
            height=cfg.capture.frame_height,
            target_fps=fps,
            mirror=cfg.capture.mirror,
        ).start() as camera, HolisticTracker(
            model_complexity=cfg.landmarks.model_complexity,
            min_detection_confidence=cfg.landmarks.min_detection_confidence,
            min_tracking_confidence=cfg.landmarks.min_tracking_confidence,
        ) as tracker:
            rep = 1
            note = ""
            while rep <= samples:
                action = _countdown(
                    camera, cfg.collector.countdown_seconds, label, rep, samples, note
                )
                note = ""
                if action is _Countdown.ABORT:
                    log.info("collection_aborted", rep=rep)
                    break
                if action is _Countdown.SKIP:
                    log.info("rep_skipped", rep=rep)
                    rep += 1
                    continue

                stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                clip_path = rec_dir / f"{stamp}_{rep:03d}.mp4"
                feats, visible_ratio = _record_clip(
                    camera, tracker, cfg.collector.clip_seconds, clip_path, fps, label
                )

                # Quality gate: no usable hands -> discard and redo this rep.
                if visible_ratio < cfg.collector.min_hands_visible:
                    clip_path.unlink(missing_ok=True)
                    rejected += 1
                    note = (f"hands seen in only {visible_ratio:.0%} of frames - "
                            "move into view, redoing rep")
                    log.warning("rep_rejected", rep=rep, hands_visible=round(visible_ratio, 2))
                    continue

                csv_path = data_dir / f"{stamp}_{rep:03d}.csv"
                np.savetxt(csv_path, feats, delimiter=",", fmt="%.6f")
                recorded += 1
                log.info(
                    "rep_saved", rep=rep, csv=str(csv_path), frames=int(feats.shape[0]),
                    video=str(clip_path), hands_visible=round(visible_ratio, 2),
                )

                # Seed the playback library with the first clean take.
                if not library_clip.exists() and visible_ratio >= 0.8:
                    library_clip.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(clip_path, library_clip)
                    log.info("library_clip_created", path=str(library_clip))
                rep += 1
    except CameraError as exc:
        log.error("camera_error", error=str(exc))
        print(f"\nCamera error: {exc}")
    finally:
        cv2.destroyAllWindows()

    print(f"\nDone - {recorded} repetitions saved for {language.upper()}:{label}"
          + (f" ({rejected} takes rejected: hands not visible)" if rejected else ""))
    if recorded:
        print(f"  landmarks → {data_dir}")
        print(f"  next step → sign-train --language {language}")
    return recorded


def main() -> None:
    """Console entry point for ``sign-collect``."""
    configure_logging()
    parser = argparse.ArgumentParser(
        prog="sign-collect",
        description="Record sign training data (landmark CSVs + videos).",
    )
    parser.add_argument("--label", required=True, help="Sign gloss, e.g. HELLO")
    parser.add_argument("--mode", "--language", dest="language",
                        choices=["asl", "psl"], required=True)
    parser.add_argument("--samples", type=int, default=None,
                        help="Repetitions to record (default from config)")
    parser.add_argument("--config", default=None, help="Path to a config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    samples = args.samples or cfg.collector.samples_per_label
    collect(args.label, args.language, samples, args.config)


if __name__ == "__main__":
    main()
