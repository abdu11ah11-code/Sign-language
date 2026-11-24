# SignBridge — Setup & Demo Guide

This guide takes you from a fresh clone to a working live demo, and is honest
about what can only be tested on your own machine (webcam, MediaPipe, and a
real training run).

---

## 1. Install

```bash
cd sign_bridge
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

First install pulls TensorFlow, MediaPipe, and OpenCV, so expect a few
minutes. Whisper downloads its weights (~140 MB for `base`) the first time you
use voice input, not at install.

**If a dependency fails to install**, the usual culprits:
- Linux microphone backend: `sudo apt install libportaudio2`
- Old pip resolving the NumPy pin slowly: `pip install -U pip` first.

---

## 2. Verify the install (no camera needed)

```bash
pytest                    # 52 tests should pass
ruff check src tests      # clean
mypy src                  # clean
```

If those pass, the whole logic layer — config, preprocessing, smoothing, word
building, tokenization, grammar, lexicons, models — is working on your machine.

Launch the UI even before training anything:

```bash
sign-bridge
```

Every tab renders in the Field Guide theme. The Sign→Speech tab will say
"No trained model yet" and tell you exactly what to collect — that's expected
on a fresh install.

---

## 3. Collect data and train (the part that needs your webcam)

The fastest path to a *visible* live demo is a tiny vocabulary. Start with
three signs plus a couple of letters:

```bash
# Whole-word signs
sign-collect --label HELLO    --mode asl --samples 40
sign-collect --label YES      --mode asl --samples 40
sign-collect --label NO       --mode asl --samples 40

# A few letters, to demo fingerspelling
sign-collect --label A        --mode asl --samples 40
sign-collect --label B        --mode asl --samples 40
```

For each sign a window opens with a 3-2-1 countdown, records ~2 s, and repeats.
Tips for a model that actually works:
- Vary your distance, angle, and lighting across the 40 takes.
- Keep the sign inside the frame; watch the skeleton overlay confirm tracking.
- If several people are available, collect from all of them.

Then train:

```bash
sign-train --language asl --architecture lstm
```

You'll see 5-fold CV scores, then a final fit, then a saved model under
`models/` with training curves, a confusion matrix, and per-class F1. The
`asl_latest.json` pointer now exists, so the UI picks the model up
automatically.

> Want to compare models? Run
> `sign-train --language asl --architecture transformer` and look at the two
> `*.metrics.json` files.

---

## 4. Demo choreography (≈3 minutes)

A clean sequence to show both directions and the accessibility story.

**Scene 1 — Sign → Speech (60s)**
1. Open the app, **Sign → Speech** tab. Note the live skeleton overlay and FPS.
2. Sign **HELLO**. Watch the focal word grow and the confidence gauge fill;
   when it crosses the threshold for ~10 frames it locks (green full stop).
3. Sign **YES**, then drop your hands for ~1.5 s. The sentence finalizes and
   drops into the transcript.
4. Toggle **Auto-speak** in the sidebar and repeat — it speaks each finalized
   sentence aloud.

**Scene 2 — Speech → Sign (45s)**
1. Switch to **Speech → Sign**.
2. Type `hello yes` and press **Translate**. The gloss line shows the adapted
   order; the player shows the stitched clips (or a storyboard if you haven't
   recorded every clip yet), with synced captions.
2. Type a name like `ABE` to show **fingerspelling** fallback for unknown
   words.
3. Click **Record voice** and say "hello" — Whisper transcribes locally and
   feeds the same pipeline.

**Scene 3 — Accessibility (30s)**
1. In the sidebar, flip **High-contrast mode** — the whole interface and the
   landmark overlay switch to maximum contrast.
2. Drag **Text size** up — the focal sign and entire UI scale together.
3. Point out **Captions always on** and that every audio cue has a visual
   twin.

**Scene 4 — PSL (15s)**
1. Switch the sidebar language to **PSL**. The masthead updates.
2. Note that PSL uses a two-handed alphabet and ships Urdu glosses — the novel
   contribution. (Live PSL recognition needs PSL data collected the same way
   as Scene 3.)

---

## 5. What only your machine can verify

Everything in this project is logic-tested and the models are validated on
synthetic data here. These items can only be exercised on real hardware, so
treat this as your acceptance checklist:

- [ ] Webcam opens and the threaded capture shows a smooth preview.
- [ ] MediaPipe draws the 21-point hand skeleton on your hands.
- [ ] `sign-collect` writes both a CSV (in `data/`) and an MP4 (in
      `recordings/`), and seeds `assets/sign_videos/asl/HELLO.mp4`.
- [ ] `sign-train` completes, reports CV accuracy, and writes a model +
      confusion matrix.
- [ ] Live recognition locks signs and builds sentences in the UI.
- [ ] TTS speaks (gTTS needs network; switch to pyttsx3 for offline).
- [ ] Microphone records and Whisper transcribes.
- [ ] Recognition quality is acceptable — if not, collect more/diverse data
      before drawing conclusions.

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "No trained model" persists after training | Check `models/asl_latest.json` exists; confirm you trained the language selected in the sidebar. |
| Camera black or "unavailable" | Another app holds the camera, or wrong `camera_index` in `config.yaml`. |
| Very low FPS | MediaPipe is the bottleneck; set `landmarks.model_complexity: 0` in `config.yaml`. Recognition still works at ~10 fps because the motion window is time-based. |
| Signs never lock | Watch the *Top candidates* bars: if the right sign is ranked first but below the ochre tick, lower `confidence_threshold`; if the wrong sign wins, record more varied clips (30+ per sign) and retrain. |
| TTS silent | gTTS needs internet; switch the sidebar voice engine to "Offline". |
| Mic errors on Linux | `sudo apt install libportaudio2`. |

---

## 7. Where to go next

- Add signs and languages: see [CONTRIBUTING.md](CONTRIBUTING.md).
- Understand the ML choices: see [ARCHITECTURE.md](ARCHITECTURE.md).
- Read the ethics and limitations before any real-world use:
  [README.md](README.md#ethics--responsible-use).
