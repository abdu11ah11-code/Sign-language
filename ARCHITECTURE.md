# Architecture deep dive

This document explains *why* SignBridge is built the way it is. For setup and
usage, see the [README](README.md).

## The 162-dimensional feature

MediaPipe Holistic emits 543 landmarks per frame (468 face + 33 pose + 21×2
hands). SignBridge deliberately uses only **162** of them:

```
[  0: 63]  left hand   21 × (x,y,z)
[ 63:126]  right hand  21 × (x,y,z)
[126:162]  upper pose  12 × (x,y,z)   nose, eyes, mouth, shoulders,
                                       elbows, wrists, hips
```

**Why drop the face mesh?** The 468 face landmarks quadruple the feature size
and the data needed to fit it, while most lexical signs are distinguished by
hand shape, location, and movement — not fine facial detail. Facial *grammar*
(e.g. question marking) matters in fluent signing and is a documented
limitation, not an oversight; it's deferred to keep the prototype trainable
from a few hundred clips per class.

**Why keep 12 pose points?** Hand position *relative to the body* is often the
only thing separating two signs with identical handshapes (a hand at the chin
vs. the chest). The shoulders also provide the reference frame for
normalization.

**Why zero-fill missing hands instead of dropping frames?** A constant-shape
window lets the model learn "hand absent" as a meaningful state (many signs
are one-handed) and keeps the temporal tensor rectangular.

## Normalization: translation- and scale-invariance

Raw MediaPipe coordinates depend on where the signer stands and how far they
are from the camera. Per frame, SignBridge:

1. **Centers** every detected point on the chest (shoulder midpoint), and
2. **Scales** by shoulder width.

This makes a sign look the same whether the signer is near or far, left or
right of frame. Zero-filled points are left at zero so they remain
distinguishable from real coordinates near the origin. The z-axis is centered
and scaled along with x/y for simplicity; because MediaPipe's z is on a
different and noisier scale, x/y carry most of the signal.

A second, optional `StandardScaler` (fit on training frames, saved alongside
the model) applies a final zero-mean/unit-variance pass. Geometric
normalization does the heavy lifting; the scaler mainly stabilizes
optimization and is part of the saved-artifact contract.

## Windowing: 2 seconds, resampled to 30 frames

Signs are motion, not snapshots. Each training clip is **2 s of wall-clock
motion** (`collector.clip_seconds`), resampled to **30 frames**. Live
recognition uses the identical recipe: the last `inference.window_seconds`
(2 s) of frames, resampled to 30. Because both sides are defined in *time*,
the model sees signs at the same speed whether the laptop runs MediaPipe at
30 fps or 10 fps. The model runs **every 2nd frame** using
`predict_on_batch` (~3 ms per call, vs ~60 ms for `model.predict`).

Brief tracking dropouts (a hand missing for under
`inference.hand_grace_seconds`, 0.4 s) are kept in the window as zero-filled
frames — exactly as in the recorded clips — instead of resetting the
recognizer. When the hands return after a real absence, the window restarts
so predictions are never made on mostly-empty frames.

## Why two model architectures

Both consume `(30, 162)` and emit a softmax over the language's vocabulary.

- **LSTM** (default): `LSTM(128) → Dropout → LSTM(64) → Dense(64) → Dropout →
  softmax`. Recurrent models are a natural fit for short motion sequences,
  train well on small data, and are the project's baseline. ~200 K params.
- **Transformer** (alternative): linear projection → sinusoidal positional
  encoding → 2 pre-norm encoder blocks → global average pool → softmax.
  Self-attention can capture long-range dependencies within the window and
  sometimes generalizes better on motion-heavy signs. ~165 K params.

Keeping both lets you compare on *your* data rather than trusting a single
choice. Select with `sign-train --architecture {lstm,transformer}`. The
transformer's positional-encoding layer is registered as a Keras serializable
so saved models reload through the normal `load_model` path.

## From raw predictions to locked signs

A per-window classifier is jittery. Two stages tame it:

1. **Temporal smoothing** (`PredictionSmoother`): a sliding history of recent
   `(label, confidence)` pairs. A sign is "stable" only when one label holds a
   ≥70% majority of the buffer *and* its mean confidence clears the threshold
   (default 0.80) across `lock_frames` (default 10) *frames* — converted to a
   number of predictions using the prediction cadence.
2. **Hold-to-confirm + cooldown** (`WordBuilder`): a stable sign is committed
   once, then a cooldown prevents the same sustained hold from registering
   repeatedly. Letters accumulate into a word; whole-word signs flush the
   current word; a long hands-absent pause finalizes the sentence.

This is what converts noisy frame-by-frame output into a clean, single locked
sign and readable sentences.

## Augmentation

All transforms operate in the *normalized* space (chest at origin, shoulder
width = 1), so rotation and scaling act cleanly around the origin:

- **rotate** ±15° in-plane — camera tilt,
- **scale** ±10% — distance variation,
- **mirror** with left/right landmark swap — doubles data for
  non-handedness-sensitive signs (use with care for PSL's two-handed
  alphabet),
- **temporal jitter** — drop/duplicate a few frames then resample, simulating
  signing-speed variation.

Crucially, augmentation and the scaler are fit **inside each CV fold**, never
on validation data, so cross-validation accuracy is leak-free.

## Text → sign

English is not signed word-for-word. `grammar.adapt_to_gloss` applies a small,
transparent rule set (drop articles/copula, move time words first, move
WH-words last) before tokenizing. The tokenizer greedily matches multi-word
phrases ("thank you") as single signs, then single known signs, and falls
back to fingerspelling for unknown words. Missing clips degrade to a clear
storyboard rather than failing.

## Why Streamlit, and its cost

Streamlit gets a styled, accessible, cross-platform UI quickly. The tradeoff:
it re-runs the script top-to-bottom rather than hosting a persistent render
loop. SignBridge lays out every tab first, then runs one continuous camera
loop at the end of the script that repaints placeholders in place; any click
interrupts the loop, Streamlit reruns, and the loop resumes. This avoids the
flicker of `st.rerun()`-driven bursts and keeps the other tabs usable while
the camera is on. Expect somewhat lower FPS than a native capture app. For a production real-time
experience, the same backend (`capture`, `inference`, `models`) can be driven
from a desktop OpenCV loop or `streamlit-webrtc` without changes.
