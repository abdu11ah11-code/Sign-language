# Changelog

> **AI assistance.** Parts of this project's code, notebooks and
> documentation were developed with the help of an AI assistant (Claude,
> Anthropic), with the course instructor's permission. All reported results
> come from runs performed and checked by the authors.

## 0.17.3 — light or dark landing page

- A **Light / Dark** switch at the top right of the landing page (website and in-app). The
  choice is remembered in the browser; with no choice the page follows the system setting.
  The 3D points view repaints in the new colours. Checked in a browser: switch, reload
  (kept), switch back, on both forms of the page.

## 0.17.2 — no faded welcome page after Start signing

- The welcome page stayed on screen, faded, for 4–5 s after **Start signing** while the
  signing screen loaded its models. It now sits in one container that the signing screen's
  first element replaces, and both buttons switch screens in a callback (one run, not two).
  Checked in a running app: the welcome page is gone 0.7 s after the click; **Home page**
  brings it back.

## 0.17.1 — the welcome page fills the window

- Inside the app the welcome page now fills the whole browser window: Streamlit's header,
  margins and column width are switched off for that screen only, and the page is as tall as
  the window (it was a small scrolling box with the 3D person cut off).
- **Start signing** is now on the page itself (header, under the intro, and in the links at
  the end); it presses the app's own Start signing button, which stays hidden. Checked in a
  running app: the welcome page fills a 1600×900 window and the button opens the signing tabs.

## 0.17.0 — landing page, and a welcome screen in the app

- **Landing page** (`src/sign_bridge/ui/landing.html`, built by `ui/landing.py`): a Swiss-grid
  page with a live 3D signer (drag to turn; he breathes, blinks, looks at the viewer and waves
  hello every few seconds, all posed in the browser, no recorded data), a live 3D view of the
  94 points the model reads (21 per hand, 12 on the body, 40 on the lips, drawn from the same
  pose), a to-scale 3D chart of the measured results, the limits, and
  "© 2026 Abdullah. All rights reserved."
- **Two forms from one template.** `docs/index.html` is the public page for GitHub Pages
  (three.js from a CDN, **Open the app** button to `http://localhost:8501`); rebuild it with
  `python -m sign_bridge.ui.landing docs/index.html`. Inside the app the same page runs offline
  with the app's own three.js and model, in the app's light theme.
- **The app opens on a welcome screen**: the landing page, with **Start signing →** at the top.
  **Home page** in the sidebar comes back to it.

Tests: 199 passing (new: both page forms, offline in-app page, measured numbers on the page,
`docs/index.html` matches the template, the app opens on the welcome screen).

## 0.16.0 — two models answer together (runs B and C)

The app now averages the answers of run B's and run C's Conv-Transformers
(`models/ensemble.py`). `models/asl_latest.json` names both under `"ensemble"`; `"stem"` still
names run C, so tools that need one model keep using it. Each model keeps its own scaler.

Measured on the **2,703 ASL Citizen test clips** (11 signers never seen in training), equal
weights decided before looking (the best weight on these signers would overstate the gain):

| | offline top-1 | through the live recognizer: correct / wrong / not sure |
|---|---|---|
| run B | 91.64% | 87.4% / 5.3% / 7.3% |
| run C | 92.64% | 88.6% / 5.2% / 6.2% |
| **B + C** | **93.23%** | **89.5% / 4.0% / 6.6%** |

The live numbers come from a new replay of all 2,703 clips (each played as 2.5 s at 12 fps,
0.5 s before and 2 s after with hands down, settings 0.35/0.05); they are **not comparable**
with the 81.0% of 22 Nov, which used another replay of 400 clips. Not yet measured on Kaggle's
test signers.

Also tried, and **not** used: averaging over trimmed copies of each clip (run C 92.64% → 92.64%,
B + C 93.23% → 92.86%) and over a mirrored copy (B + C 93.27%, +0.04 for twice the work).
Re-tuning the "not sure" bar for B + C (chosen on 6 of the 11 test signers, checked on the
other 5) found nothing better than the current 0.35/0.05.

Cost: one guess takes 14 ms instead of 6 ms (cloud CPU); the camera loop is unaffected.
The status line under the transcript says "2 models together".

Kaggle notebook: **run D** (`notebooks/train_asl_run_d.ipynb`) is run C's recipe with another
training seed (`SB_TRAIN_SEED`, 7). The signer split and the live-check clips keep seed 42, so
every run is scored on the same people. With earlier models attached as an input, a new cell
scores each one and every average of them on Kaggle's test signers (and ASL Citizen's), and
sends the best combination through the live check; results go to `ensemble_results.json`.

Tests: 192 passing (new: averaging and weights, each member's own scaling, refused
ensembles, the live recognizer on an ensemble, loading one from `asl_latest.json`, run D's
notebook).

## 0.15.0 — the 3D person looks more alive, and run C is the app's model

The 3D person, polished in the page that poses it (`ui/figure3d_person.html`); the model file
`person.glb` is unchanged.

- **The hand that is not signing rests at the side**, elbow back, fingers loosely bent. Before,
  it followed the body tracker's guess for a wrist that was out of the picture, so it stuck
  out sideways with stiff, spread fingers. A wrist guessed more than 0.6 shoulder-widths below
  the shoulders, with the hand itself unseen, now counts as resting. Moving between signing
  and resting eases over about 0.15 s instead of jumping.
- **Eyebrows and eyelids follow the recorded face**: brows rise (as in a yes/no question),
  knit (a frown, as in a *wh-* question) and the eyes narrow when the signer's did. The
  neutral values are the middle of the library's 6,000 frames (brow 0.126 of the face height
  above the lid, eye opening 0.216 of its width, inner brows 0.192 of the face width apart);
  small differences are ignored, because they are mostly the signer's own face.
- **He blinks** every 2–5 s, and **his eyes look at the viewer** (up to about 11 degrees from
  straight ahead, so they never roll).
- **Steady head lean is eased.** The library's recordings lean the head back or forward by up
  to 40 degrees (median −3), mostly from camera height and the tracker's depth guess: SAD and
  HELLO looked up at the ceiling. Each sign's steady lean is now eased to at most about
  10 degrees; nods within a sign are kept, and in a sentence the change eases between signs.
- Shirt changed from black to charcoal so its folds and the arm's shape show.

Also: run C's model (`asl_conv_transformer_20261123_222845`) is installed as the app's model
(`asl_latest.json`). Kaggle test top-1 72.26% (run B 68.98%); ASL Citizen test 92.64%. See
the results log. Run B's pointer is kept in `models/run_b_20261122/`.

Tests: 185 passing (new: head lean measured and eased, sentence easing, expression points,
face bones present).

## 0.14.1 — signs with movement: a guess while you move, fewer "not sure"

Abdullah saw signs such as HELLO, MOM and ALLIGATOR recognised only after his hand stopped.
Measured first, on webcam-style ASL Citizen clips (people raise their hands, sign, lower
them), streamed frame by frame through the app's own recognizer at 12 fps with the v0.4
model (each clip timed as 2.5 s):

- The model **does** use the whole movement: 95% of clips became exactly one segment, and the
  segment's best guess was right 67% of the time (70% when the model is given the whole clip
  directly). The answer just waits until the hands have been still for `hold_seconds`.
- The real loss was the **confidence bar**: tuned on Kaggle's tightly cut clips, it turned
  many right answers into "not sure".

Changes:

- **A guess while the sign is still moving**: every 0.4 s of signing, the movement so far is
  classified and shown faded ("Mom… — watching the movement, looks like Mom (62%) so far").
  It is never written down; the answer still comes when the sign ends. One extra model run
  (~30 ms) every 0.4 s while signing.
- **Live settings re-measured** (chosen on 400 val-signer clips, then checked once on 400
  test-signer clips): `segment_confidence` 0.45 → **0.35**, `segment_margin` 0.10 → **0.05**,
  `hold_seconds` 0.6 → **0.4**. Test signers: correct **49.2% → 54.0%**, not sure 40.8% → 34.0%,
  wrong 10.0% → 12.0%. The answer comes 0.2 s sooner; shortening the hold did not change accuracy
  or split more signs in two (6.0% of test clips either way).
- The training and tuning notebooks' live checks count only the answers given when a sign
  ends (live guesses switched off there), so their numbers stay comparable.
- Fixed: `reset()` now clears the live guess. Tests 179 → 182.

## 0.14.0 — run C: train the best model longer and harder

Prepared while run B trains. Nothing here is a result yet.

- `notebooks/train_asl_run_c.ipynb` (built from the same cells as the main notebook, only the
  settings differ): **conv-transformer only**, **200 epochs** (run A's best epoch was 93 of 100,
  still improving), **stronger augmentation** and **2-frame motion**. Attach the same inputs as
  run B.
- Stronger augmentation (`SB_STRONG_AUG=1`): besides flip/rotate/scale/stretch/time-crop, a hand
  (12 %) or the lips (10 %) are hidden now and then, and half the time a short run of frames
  (up to 12 %) is blanked — what happens on a webcam when tracking drops out. Hidden points are
  exactly zero, as the live tracker reports them.
- `TemporalDelta(lags=(1, 2))`: the model also sees the change over two frames. Models saved
  before keep `lags=(1,)` and load unchanged; the app needs no change (same 188 inputs).
- Settings now read `SB_ARCHS`, `SB_STRONG_AUG`, `SB_MOTION_LAGS`; the main notebook's defaults
  are unchanged (runs A and B stay reproducible). Run C checked end to end on the synthetic
  dataset. Tests 177 → 179.

## 0.13.1 — the My signs box stays open

- Removing (or adding) a sign closed the **My signs** box after every click. The box's
  title held the count ("My signs — 3 of 250"), and Streamlit draws an expander whose title
  changed as a new, closed box. The title is now always "My signs", the count is shown
  inside it, and once you start choosing the box stays open until you close it yourself.
  Checked in a browser: add MOM, DAD, DRINK, remove one with its × — the box stays open
  ("Recognizing 2 of 250 signs").

## 0.13.0 — speak, and the 3D person signs it

The Speech → Sign tab needed recorded sign *videos* (a fresh install has none) and recorded
from the PC's microphone for a fixed 15 s through `sounddevice`. Now:

- **Mic in the browser** (`st.audio_input`): press the mic, speak, press stop. The recording
  is decoded (`wav_to_whisper`: mono, 16 kHz, float32 — no ffmpeg needed) and written down
  by Whisper on this computer; the text appears in the box, which can still be typed in.
- **The 3D person signs the sentence.** `inference/text_to_motion.plan_motion` plans it
  against the animation library's signs (250 for ASL): the gloss itself, multi-word signs
  (*thank you* → THANKYOU, *ice cream*), everyday synonyms (*mother* → MOM, *television* →
  TV, *my* → MINE/MY), and plain English endings (*dogs*, *drinking*, *cried*); ASL word order
  as before (time words first, question words last, no *the*/*is*). Words with no sign are
  shown crossed out and listed under the figure.
- `figure3d.join_clips` joins the signs into one clip, moving smoothly from each sign into
  the next and resting after the last; `sign_3d_html(..., segments=)` gives **every sign its
  own touch spot** and shows the **word being signed** ("DRINK 4 / 5") above the person.
  Mirror is off by default here (the person signs *to* you), with a Slow switch.
- The older video/storyboard path is kept for when the animation library or the 3D person
  is not installed. The "Speak" button on the Sign → Speech tab is unchanged: it reads the
  signed sentence aloud.
- Checked in the running app: typed "Hello, my mother is drinking water and I am a human" →
  HELLO MINE/MY MOM DRINK WATER, with *and, I, human* marked as having no sign; the person
  signs it with the thumb spot on DRINK. Tests 169 → 177.

## 0.12.0 — ASL Citizen joins training (run B)

**Measured first** (`sign-check-aslcitizen`, model `asl_conv_transformer_20261120_122420`, run on
Abdullah's PC on 22 Nov 2026): on the 5,119 imported ASL Citizen clips whose gloss is exactly one
of the model's signs, the current model scores top-1 72.6% / top-5 89.8% (train signers),
75.2% / 91.0% (val), 70.2% / 88.4% (test). These are people and cameras it never saw, and the
scores are close to its 68.54% on unseen ASL Signs signers. The check kept 228 of 260 ASL Citizen
glosses (all 171 exact names and 57 variants/aliases) and dropped 32 variants that are a
different sign filed under the same English word (e.g. CRY1–4, HELICOPTER1–3, CAT2, CAN2).

- `asl_citizen.training_clips(folder, labels, splits, kept_only=True)`: the imported clips of
  the chosen ASL Citizen splits, labelled for a model, only for glosses the check kept.
- **Training notebook**: if an ASL Citizen import is attached (a folder with `manifest.csv`
  and `check.json`), its *train + val* signers are added to **training** (4,121 clips for 204
  signs with the current import). Validation and the main test stay the same ASL Signs signers,
  so scores compare directly with run A; ASL Citizen's own *test* signers are reported as a
  second, independent test (`aslc_top1`, `aslc_top5` in the table and in `*.metrics.json`).
  The scaler is fitted on the combined training clips. Without the import, the notebook
  behaves exactly as before.
- Checked by running every notebook cell end to end on a small synthetic ASL Signs + ASL
  Citizen dataset; tests 165 → 169.

## 0.11.0 — a real 3D person signs

0.10.0 wrapped the portrait over the old mannequin's head; it looked like a mask
and was rejected. This release replaces the figure with a real 3D person.

- **The person** (`ui/static/person.glb`, 4.2 MB) was built in Blender 5.3 with
  MakeHuman's CC0 data: a young adult man's body, the 163-bone "default"
  skeleton (every finger joint, jaw, eyes, face) and its skin weights, real
  eyeballs. The face is the author's Gemini portrait (AI-generated, not a real
  person): MediaPipe Face Mesh landmarks on the portrait and on a render of the
  head drive a thin-plate-spline warp, the portrait's own lighting is taken out,
  and it is baked into the head's texture with a skin tone sampled from the same
  portrait. Hair (3,200 cards combed back along the head), a dark long-sleeved
  shirt, trousers and shoes are generated by the scripts. Everything is
  rebuildable with `tools/avatar/build_all.sh`; sources and licences are in
  `tools/avatar/README.md` and `ui/avatar/README.txt`.
- **Posed from the recording every frame** (`ui/figure3d_person.html`): torso
  turn from the shoulders, head and neck from the face (a little calmer than
  recorded), jaw from the inner lips, arms by two-bone IK with the elbow on the
  recorded side, each hand turned to the recorded palm (the forearm takes half
  the twist so the wrist does not wring), and each finger bone pointed along the
  recorded joints. Near the face, hand positions are measured against the face,
  so they land on the person's face, not just near it.
- **The red touch spot** now sits on the person's skin, found through 152 face
  points located on the person (rays from the front view through the landmarks);
  during the touch the arm is re-solved so the fingertip rests on the spot.
  Checked in a browser: DAD (thumb on forehead), MOM (thumb on chin), DONKEY
  (thumb at the temple), YES (fist, no spot), TV (V hand) inside the running app.
- **Served by the app**: `sign-bridge` and `.streamlit/config.toml` turn on
  Streamlit static serving; the page fetches `/app/static/person.glb` from the
  app's own server (nothing from the internet). three.js' GLTFLoader (MIT,
  r160) is bundled as a classic script. Without the model the old mannequin is
  drawn; without WebGL, the flat drawing.
- Removed the 0.10.0 photo-mask look (`person_look`, `avatar/face.json`); the
  portrait moved to `tools/avatar/portrait_ref11.jpg` as a build source.
- Tests: 161 → 165.

## 0.10.0 — the 3D figure looks like a person

The plain mannequin made it hard to see *where on the face* a sign happens. The
figure now wears a real-looking face, so eyes, nose, lips and hairline are the
landmarks a learner already knows.

- **Face from a portrait.** `ui/avatar/face.jpg` is the front portrait from a
  character reference sheet the project author (Abdullah) made with Google
  Gemini — AI-generated, not a photo of a real person (see
  `ui/avatar/README.txt`). MediaPipe Face Mesh (0.10.14) was run on the
  portrait to find where each of the figure's 152 face points sits in it
  (`face.json`), so the picture is wrapped over the moving face surface: the
  portrait's mouth opens and brows lift with every recorded sign.
- **Skin, hair and shirt colours** sampled from the same portrait; a hair cap
  over a skin-coloured head, a dark shirt on the torso and arms.
- The face surface now covers the eyes when the photo is used
  (`face_triangles(open_eyes=False)`); the drawn feature lines and pupils are
  hidden, since the photo has its own.
- **Close-up button** zooms smoothly onto the head (still turnable), so the
  red touch spot — forehead for DAD, temple for DONKEY — is easy to see.
- `sign_3d_html(..., person=False)` keeps the plain mannequin; it is also used
  automatically if the avatar files are missing.
- The caption under the 3D view no longer says "Green = hands" (that legend now
  shows only under the flat drawing).
- Tests: 157 → 161.

## 0.9.2 — a red spot shows where the finger touches the face

The baseline live test found that **placement decided recognition** (DAD 0/3 at
the temple, 3/3 on the forehead), and the 3D figure still did not *say* where
the thumb goes: on MOM the hand floated in front of the chin.

- **`find_touch()`** finds, from each recording, the fingertip that touches the
  face. From the front every fingertip of a hand before the face looks inside
  it, so two things decide: depth *within the hand* (MediaPipe's finger depth
  relative to the wrist is reliable; the touching tip is pushed furthest
  towards the face) and sitting right on a face feature (nose tip, lips, chin).
  Positions are measured in the face's own coordinates, so a turned head does
  not skew them.
- **Checked on the real library against signs with known answers:** MOM and
  GRANDMA → thumb at the chin; DAD, GRANDPA, MAN → thumb at the forehead;
  DONKEY → thumb at the temple (so DAD and DONKEY point to visibly different
  places); NOSE, LIPS, TOOTH, THINK → index finger; YES and AIRPLANE → no
  touch. Touch points found for 170 of 250 signs.
- **Shown in 3D** as a large red spot with a white ring and a pulsing halo,
  fixed to the face triangle under the fingertip, plus a red marker on the
  fingertip and a "Thumb here" label. During the touch the hand is moved so the
  fingertip rests on the spot — the touch fixes the hand's depth, which the
  recording only estimates.
- **Not done, on purpose:** naming the region ("chin", "temple"). Tried three
  ways; heads in the recordings are often turned and the name flipped between
  forehead and temple for DAD — a wrong word would teach the wrong place. The
  spot shows the place itself.
- Tests: 154 → 157.

## 0.9.1 — fixes found on the first real 3D library

Checked on the real version-2 library from Kaggle (250 signs, built 21 Nov from
0.9.0): all 250 load, draw in 3D and flat, and have a face surface; depth sits
mostly between the chest and 17 units in front of it.

- **Fixed: untracked hands were drawn as a blob at the top corner.** Visibility
  flags came from a 0/1 array, and three.js hides an object only when
  `visible === false` — a `0` still draws. Every flag is now a real boolean,
  with a test that keeps it so.
- **Fixed: the torso twisted and leaned back.** MediaPipe's shoulder and hip
  depth is noisy (one shoulder 8 units forward, the other 8 back, on DAD). The
  figure keeps a third of it, which works on the library already built; the
  notebook now also limits it to ±3 for future builds.

## 0.9.0 — a 3D figure with a face, instead of the stick figure

The "How to sign" box now shows a **solid 3D figure you can turn with the
mouse**: a face whose mouth opens and eyebrows lift as recorded, jointed hands
(finger bones, knuckles, palm), arms, neck and torso. Front / Side / Angle
buttons; double-click resets. A **3D** switch returns to the flat drawing.

Why it matters beyond looks: the counted live test (21 Nov) found that **hand
placement decided recognition** — DAD went 0/3 → 3/3 by moving the thumb from
the temple to the forehead. Placement in depth is exactly what a front-on stick
figure cannot show and a turnable one can. And ASL uses the face as grammar
(raised eyebrows mark a yes/no question), which the old figure had no face for.

- **Library version 2** (`notebooks/make_sign_animations.ipynb`, re-run on
  Kaggle): every point keeps **x, y, z**, and the face adds its outline, eyes,
  eyebrows, nose and lips — 152 of MediaPipe's 468 face points, the exact sets
  taken from MediaPipe's own `face_mesh_connections`. 206 points per frame.
  Version-1 libraries still load and draw flat; a version-2 file whose face
  points differ from this release's is refused rather than drawn wrong.
- **One depth space.** MediaPipe measures each hand's depth from its own wrist
  and the face's from its own centre; both are re-anchored on the body's
  wrists and nose.
- **Arm depth from bone lengths.** MediaPipe's arm depth put the hands ~40
  units in front of the body — arms fully extended — on real clips. Now only
  its *direction* is used: an upper arm that looks short from the front points
  at the camera by exactly what Pythagoras gives, from typical bone lengths
  (upper arm 21, forearm 18, in a 26-unit shoulder width).
- **Face surface.** The face points are triangulated once on the clip's
  average face (Delaunay), leaving the eyes and the mouth open, so the surface
  moves with the face; a head shape behind it, eyes with pupils, brows, gold
  lips, a dark mouth inside.
- **Offline.** three.js r160 (MIT, licence file included) is bundled in
  `ui/vendor/`; the page fetches nothing. If WebGL is unavailable the flat
  drawing is shown instead.
- Uses `st.iframe`, which replaces `st.components.v1.html` (deprecated in
  Streamlit 1.60 — "will be removed after 2026-06-01"); older Streamlit falls
  back to the old call.
- **Fixed (old bug):** the sign name was not HTML-escaped in the flat
  drawing's label.
- **Verified here, not on real competition data yet:** the whole notebook was
  run on Kaggle-format files built from 6 real ASL Citizen clips (real hands,
  arms and lips with real depth) with MediaPipe's canonical face fitted to the
  real lips; the figure was rendered in a real browser and inside Streamlit,
  front, side and angled, animated and dragged. The real check is the
  notebook's `animation_preview.png` on the competition data.
- Tests: 141 → 153.

## 0.8.1 — signs ASL Citizen names differently, checked by the model

The first full ASL Citizen import matched **208 of the model's 250 signs** by
name. Several of the 42 "missing" were not missing but named differently:
ASL Citizen follows the ASL-LEX dictionary, Google's list a children's
vocabulary — DAD is FATHER there, MOM is MOTHER, ALLIGATOR is CROCODILE.

- **`--aliases`** on `sign-import-aslcitizen` also imports 25 such candidates
  (FATHER → DAD, MOTHER → MOM, WE + US → WEUS, …). A candidate that is itself
  one of the wanted signs is never re-filed (AWAKE stays AWAKE). Re-running an
  existing import with `--aliases` adds only the new videos.
- **New `sign-check-aslcitizen`** decides which candidates to keep, by
  measurement rather than by trusting English synonyms: the trained model is
  run on each candidate's videos, and a candidate is kept only if its top-5
  agreement reaches the 10th percentile of the signs that matched exactly. The
  same test catches numbered variants the importer merges (NUT1 + NUT2 → NUT),
  which can be two different signs. Exact matches are always kept.
- The decision uses ASL Citizen's **train and val signers only**, so its test
  signers stay untouched for a webcam benchmark. The check also reports the
  current model's accuracy on ASL Citizen per split.
- **First measurement, 20 trial clips (all one sign, AFTER):** the v0.4 model,
  which never saw ASL Citizen, got top-1 **2/20** and top-5 **8/20** (chance:
  0.4% / 2%), with low confidence — well above chance, so the preparation is
  right, but weak on webcam video. Too small to report; the full check gives
  the real figure.
- `load_model_from_stem()` in the evaluator loads any saved model, not only
  the latest one.
- Tests: 127 → 141.

## 0.8.0 — webcam training data (ASL Citizen importer)

The measured 38-point gap between benchmark accuracy (69.1%) and live accuracy
(31.0%) has two causes. v0.7 fixed the segmentation half of it. The other half
is the data: Google's ASL Signs clips were filmed on **phones held close to the
signer**, while the app watches a **laptop webcam** from across a desk. The
remaining live bottleneck is now the model's own confidence — 47.2% of signs
come back "not sure" — which is a training-data problem, not a tuning one.

- **New `sign-import-aslcitizen`**: converts Microsoft Research's ASL Citizen
  (83,399 **webcam** videos, 2,731 signs, 52 Deaf signers) into the landmark
  clips the trainer reads, through the same `mediapipe==0.10.14` Holistic
  pipeline the live app uses.
- **Reads straight out of `ASL_Citizen.zip`** — each wanted video is extracted
  to a temporary file, tracked and deleted, so the 42.8 GB download needs no
  second 42.8 GB beside it.
- **Runs locally, not on Kaggle**, by necessity: the zip is bigger than a
  Kaggle session's disk, and `mediapipe.solutions.holistic` was removed after
  0.10.21, which itself needs Python ≤ 3.12 while Kaggle now runs 3.13.
- **`--workers` tracks several videos at once** (default: one per core, minus
  one), which is what decides whether the run takes hours or most of a day:
  tracking measured at **≈30 ms a frame**, so one video at a time would put a
  7,500-video import past five hours.
  - Threads, not worker processes. MediaPipe releases the GIL while its C++
    graph runs, measured on this pipeline: two threads on two cores tracked
    **1.9x** as many frames a second as one (11.5 → 5.9 ms/frame), and a whole
    import of the same videos ran in **1.0 s against 1.6 s**. Processes were
    tried first and rejected: with the `spawn` start method — the only one
    Windows has — every worker re-imports the program, which **hung** a script
    without an `if __name__ == "__main__"` guard and cannot work from a
    notebook at all.
  - The zip is read under a lock, since one archive means one file handle that
    seeks as it reads. Pulling a video out takes milliseconds against seconds
    of tracking, so the threads barely queue for it.
- **Resumable**: Ctrl-C and re-run the same command; anything already in
  `manifest.csv` is skipped. Unreadable videos are counted and reported, never
  fatal — including ones that fail while being read out of the zip, which a
  test caught escaping the error handling and killing the whole run.
- `--labels` keeps only the signs a model already knows (matching glosses
  case- and punctuation-insensitively, so `ice cream` = `ICE_CREAM` =
  `ice-cream1`), and `--per-sign` spreads its pick across signers.
- Tests: 107 → 127. The new ones build a real miniature dataset — real `.mp4`
  files, real split CSVs, zipped the way the release ships — and one checks
  that four threads write byte-identical clips, manifest and report to one
  thread, in the same order.
- **Licence note, in the module, the README and the CLI's own help:** ASL
  Citizen is for non-commercial research only and may not be redistributed.
  Nothing derived from it is committed; `data/` stays git-ignored.

## 0.7.1 — no more doubled words

- **The same sign reported twice in quick succession is written down once.**
  Live testing kept producing `DAD · DAD · NOT · DRINK · DRINK`: one sign cut
  into two pieces, each recognized separately. A sign matching the previous one
  within `repeat_window_seconds` (1.0 s) is now treated as the same sign
  continuing. After a real pause, a genuinely repeated sign still counts twice,
  since signers do repeat words for emphasis.
- This complements v0.7.0's `merge_gap_seconds` (0.3 s), which joins the pieces
  before the model sees them; this catches the longer pauses that slip past it.
- Tests: 105 → 107.

## 0.7.0 — live segmentation, measured and fixed

Measured with `tune_segmenter.ipynb` on 800 clips from signers the model never
saw, then applied.

- **Sign length limits were wrong at both ends.** Real signs run **0.2–9.4 s**
  (median **0.5 s**), but the app only accepted 0.3–3.0 s, discarding short
  signs as twitches and cutting long ones in half. New limits: **0.05–6.0 s**.
  Measured effect: signs missed entirely fell from **17.7% to 0%**, and
  end-to-end correct answers rose from **30.2% to 35.0%**.
- **A pause mid-sign no longer splits it in two.** Signers often hesitate in the
  middle of a sign; the segmenter now waits `merge_gap_seconds` (0.3 s) before
  reporting, and treats movement resuming within that window as the same sign
  continuing. Dropping the hands still reports immediately, with no delay.
- Also tuned from the measurement: `motion_stop_speed` 0.25 → 0.15 and
  `hold_seconds` 0.35 → 0.6.
- **Honest note:** with the new limits, wrong answers rose from 12.6% to 17.8%.
  Much of that looks like one sign being reported as two, which is what the
  merging above is meant to fix — the next run will show whether it did.
  `motion_start_speed` could not be measured by this test (each test clip begins
  with the hands entering the frame, which starts a sign whatever the threshold),
  so it is unchanged and still matters live.
- Tests: 102 → 105.

## 0.6.2 — the segmenter sweep, made usable

- **`SignSegmenter.update` accepts an already-measured hand speed.** The speed
  of a hand depends only on the recording, never on the settings, so an offline
  sweep can measure it once per clip instead of once per setting. Live callers
  are unchanged. A test proves both paths produce identical segments.
- **`tune_segmenter.ipynb` now finishes in minutes** instead of running for over
  half an hour, and it sweeps **`min_sign_seconds` as well as
  `max_sign_seconds`**. That matters: measured on real test clips, half of all
  signs last **under 0.5 s** and the shortest 5% are **0.2 s**, while the longest
  reaches **9.4 s** — so the shipped limits of 0.3 s and 3.0 s were discarding
  short signs as twitches and cutting long ones in half.
- Tests: 101 → 102.

## 0.6.1 — measuring the live gap

- **New Kaggle notebook `tune_segmenter.ipynb`.** The v0.4 run showed the app
  recognising only 31% of real test clips while the same model scores 69% when
  handed the clips directly — most of the loss happens in deciding where a sign
  starts and ends. This notebook measures that step instead of guessing:
  - **Stage 1** streams real clips past the segmenter alone (no model, so it is
    fast) and scores ~288 candidate settings on *did it find exactly one sign,
    and did that sign line up with the real one?*
  - **Stage 2** runs the best few settings through the whole pipeline, model
    included, for correct / wrong / not sure / missed.
  - It prints the winning `config.yaml` block, and says plainly when the current
    settings are already as good as anything tried.
- No behaviour change in the app itself; this release is the measuring tool.

## 0.6.0 — "How to sign" animations

- **A looping stick figure that shows how a sign is made**, in a new box under
  the live camera. Built from real recordings by Deaf signers, drawn as hands,
  arms, shoulders and lips — no video, no face, nobody identifiable.
  - Pick any sign from your **My signs** list (or all of them).
  - **Speed:** normal, slow (half speed) or **step**, with a slider to move
    frame by frame through a hard sign.
  - **Mirror** (on by default) flips left↔right, so you copy it as in a mirror.
  - Green = hands, gold = lips.
- **Where the animations come from:** `notebooks/make_sign_animations.ipynb`
  runs on Kaggle (CPU, a few minutes) and picks, for each of the 250 signs, the
  recording **closest to the average** of that sign — a typical example, not an
  unusual one. Clips where a hand was lost are skipped; a hand that disappears
  briefly keeps its last position instead of jumping to the corner.
- **Licence:** the library is derived from competition data, so
  `assets/sign_animations/` is git-ignored. Keep it local; do not redistribute.
- The box simply does not appear when no library is installed, so the app runs
  exactly as before without it.
- Tests: 91 → 100.

## 0.5.0 — AI sentences

- **Signs → natural English.** When a sentence is finished (hands lowered,
  or **Finish**), the signed words are sent to an AI model, which writes
  them as an English sentence: `MOM DAD NO DRINK` → "Mom and Dad don't
  drink."
  - If the signs can be read in more than one way, up to 3 readings are
    shown.
  - The AI may only add grammar, never new meaning, and the raw signs are
    always shown next to its sentence.
  - The AI sentence goes into the transcript and is what **Speak** and
    auto-speak read aloud.
- **Providers:** Google Gemini (default; free tier), Claude (Anthropic) or
  OpenAI, set with `ai.provider` / `ai.model` in config.yaml.
- **Background writing:** it runs in the background, so the camera never
  freezes while the AI writes.
- **Fallback:** without a key or internet, or on any provider error, the
  app falls back to the plain words and says why.
- **API key:** pasted once in the sidebar (**AI sentences**) and stored in
  `.streamlit/secrets.toml`, which is git-ignored. Environment variables
  (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) also work.
- **Privacy:** only the sign words are sent to the AI service, never video
  or landmarks.
- Tests: 77 → 89 (all providers and error cases faked, no network needed).

## 0.4.0 — live recognition that waits for a whole sign

- **One guess per sign (segment mode).** Whole-sign models (such as the
  Google ASL Signs ones) were being fed a sliding window and had to answer
  every moment, including while the hands rested or moved between signs.
  That showed up as a stream of random words. A new motion-based segmenter
  (`inference/segmenter.py`) watches hand speed, measured in shoulder
  widths per second:
  - movement starts a sign;
  - holding still, lowering the hands, or a time limit ends it.
  The model then runs once on the whole sign. The segment is re-timed to
  30 fps first, so a sign looks the same however fast the laptop tracks.
  Used automatically for these models (`inference.mode: auto`).
- **"Not sure" instead of a wrong word.** A finished sign is accepted only
  if the model is confident (`segment_confidence`) and clearly prefers it
  over the runner-up (`segment_margin`). Otherwise the app says "not sure"
  and shows its best guesses.
- **My signs.** Choose the signs you know or are practising, and the
  recognizer only picks among those. The choice is saved per language in
  `data/<lang>_my_signs.json`.
- **How to sign it.** Each recognized ASL sign, and each sign in "My
  signs", links to a video search showing how to sign it. Multi-word ASL
  Signs glosses (`GLASSWINDOW`, `THANKYOU`, …) now display and build
  sentences as real words.
- **Camera-shape-invariant features** (`holistic_lips_clip`). MediaPipe
  divides x by the image width and y by the image height, so a portrait
  phone video and a 4:3 webcam stretch the body differently. This mode
  normalizes per clip and rescales y using a body proportion (nose to
  shoulder line). Tested: a clip squashed to another camera shape gives the
  same features.
- **Kaggle notebook v0.4:**
  - uses the new features;
  - all models run the full learning-rate schedule (no early stopping, so
    none is cut off early), and the best validation epoch is kept;
  - new **live-pipeline check**: 1,000 real test clips are streamed frame
    by frame through the app's own recognizer, and the notebook reports
    correct / wrong / not sure / missed, plus a confidence-threshold table.
- Tests: 66 → 77.

## 0.3.0 — Transformer models, lip features, ASL Signs training

- **Two Transformer architectures** next to the LSTM, all selectable with
  `sign-train --architecture`:
  - `transformer`: pure self-attention encoder (as before).
  - `conv_transformer` (new): a 1D-convolution + Transformer hybrid in the
    style of the top Kaggle ISLR solutions. A frame-difference (motion)
    input feeds depthwise temporal convolutions with efficient channel
    attention (ECA), interleaved with Transformer blocks. It also uses
    stochastic depth, late dropout, AdamW and label smoothing.
  - The model's structure is saved next to its weights (`.spec.json`), so
    any saved model reloads even if the full `.keras` file cannot.
- **Lip landmarks.** Tracking now keeps 40 lip points from the MediaPipe
  face mesh (282 raw values per frame; old 162-value data still loads).
  The new `holistic_lips` feature mode (188 values: hands, upper body and
  lips, chest-centred and shoulder-scaled, 2D) is saved with the model and
  used automatically in the live app. Mirror augmentation also swaps the
  left and right lip points.
- **Kaggle training notebook** (`notebooks/train_asl_signs_kaggle.ipynb`):
  - Converts the Google *Isolated Sign Language Recognition* (ASL Signs)
    data with the same preprocessing code the live app uses.
  - Splits by participant, so test signers are never seen in training.
  - Trains the LSTM, Transformer and Conv-Transformer with identical
    settings and reports top-1, top-5, macro-F1, size and CPU latency.
  - Exports the models in the format the app loads.
- **Live window follows the model.** The live recognizer takes its frame
  count from the loaded model and its time window from the model's
  metadata, so a 64-frame ASL Signs model is fed exactly what it was
  trained on.
- **Scaler fix.** Features that are nearly constant after normalization
  (the shoulders) had a tiny standard deviation, which blew small wobbles
  up into huge values. Per-feature scale is now floored at 0.01. Found
  because augmented training failed to learn; with the fix it does.
- **Stability fix.** Keras' built-in `DepthwiseConv1D` crashes TensorFlow
  2.16 on CPU, so the conv blocks use a custom layer instead.
- Tests: 57 → 66.

## 0.2.1 — public PSL dataset support (UAlpha40)

- **`sign-import`**: converts a folder-per-sign dataset of photos and/or
  videos into SignBridge training clips. Photos go through MediaPipe Hands
  (with a padded retry for hands cut off at the edge) and become held
  hand-shape clips; videos go through Holistic and are trimmed to the frames
  with hands. Coordinates are converted to live-webcam units so tall photos,
  16:9 videos and the 4:3 webcam give the same hand shape. Re-runs resume.
  It also adds one clip per sign to the Speech → Sign library.
- **`hand_shape` feature mode** (`sign-train --features hand_shape`): one
  hand relative to its own wrist, scaled by its size. This is needed for
  close-up hand photos that show no body. The mode is saved with the model
  and used automatically at inference.
- **Urdu alphabet output**: the 40 UAlpha40 signs map to Urdu letters, so
  fingerspelled PSL builds Urdu words and gTTS speaks them with the Urdu
  voice. The UI shows names like `Kaf · ق`.
- Verified on a UAlpha40 subset: 96% of photos had a detectable hand. On
  held-out signers, 6 of 7 detectable letters locked correctly with no
  wrong locks, and held-out Jeem videos locked through the live tracker.
- Tests: 52 → 57.

## 0.2.0 — recognition, UI and data-quality upgrade

Every item below was reproduced before the fix and is covered by a test or
a scripted check.

### Live recognition (Sign → Speech)

- **Signs now lock despite tracking dropouts.** Previously, a hand missing
  for a single frame reset the smoother, so with MediaPipe dropping a hand
  about one frame in seven, a sign *never* locked. Dropouts shorter than
  `inference.hand_grace_seconds` (0.4 s) are now tolerated.
- **Lock time cut from ~6 s to ~2.5 s at 12 fps.** `lock_frames` was counted
  in predictions (one every 5 frames), so "10 frames" really meant 50. It is
  now converted to predictions using the prediction cadence.
- **Time-based motion window.** Live input is the last 2 s of frames
  resampled to 30, the same recipe as the training clips, so recognition no
  longer depends on how fast the laptop runs MediaPipe.
- **Fresh window when the hands return**, so the first predictions are not made
  on mostly-empty frames (fewer false locks).
- **Inference ~18× faster:** `predict_on_batch` takes ~3 ms per window,
  vs ~60 ms for `model.predict`. The model now runs every 2nd frame.
- **Top-3 candidate bars** in the UI show the live probabilities against the
  lock threshold. This helps with live debugging and with explaining
  decisions.

### Data collection (`sign-collect`)

- **Quality gate:** a take where hands were visible in under 60% of frames
  is rejected and re-recorded (`collector.min_hands_visible`).
- **Skip key works:** pressing `s` used to record the rep anyway.
- Clips are recorded by wall-clock time (`clip_seconds`), matching the live
  window.
- On-screen text no longer shows `?` (OpenCV cannot draw the em dash).

### Training (`sign-train`, `sign-eval`)

- **Transformer models can now be loaded by the app.** The custom
  `PositionalEncoding` layer was never registered at load time, so
  deserialization failed.
- **Small datasets no longer crash.** Stratified splits and CV folds adapt
  to the number of clips per sign, and you get a clear message when a sign
  has fewer than 3 clips.
- New `--skip-cv` and `--epochs` flags for quick iteration.
- `sign-eval` maps classes by name when the recorded signs and the model
  vocabulary differ. Before, metrics were silently wrong.

### Speech → Sign

- **Multi-word signs work:** "thank you" now plays `THANK_YOU.mp4` instead of
  fingerspelling, because underscore file names now match spaced phrases.
- **Phrase-safe grammar:** "good morning" stays one sign instead of being
  reordered to "MORNING GOOD".
- **Sentence pauses** are inserted at `. ? ! ,` (they used to be dropped).
- **Captioned video works without ImageMagick.** Captions are drawn with
  OpenCV. Before, the video always fell back to the text storyboard on
  machines without ImageMagick.
- Voice input now fills the text box. Before, a keyed widget ignored the
  transcribed text.
- The translation persists across reruns, Enter translates, and rendered
  videos are cached per text.
- The caption line collapses fingerspelling: `ALI (A-L-I)`.

### Interface

- **No more flicker:** a single continuous camera loop repaints in place,
  replacing `st.rerun()` every 6 frames.
- **All tabs usable while the camera runs.** `st.rerun()` used to stop the
  script before the Speech → Sign and Conversation tabs were drawn.
- **Two-way conversation log:** the hearing person's replies are logged next
  to the signer's sentences, tagged *Signer* / *Speaker*.
- Pause releases the webcam, so its light turns off.
- Offline voice (pyttsx3) runs in a subprocess. It used to hang on the
  second sentence.
- The transcript copy box uses `st.code` (with a copy button), which avoids
  duplicate and stale widget ids.
- Fixed the page title clipped under the toolbar and button labels breaking
  mid-word. The accent colour now matches the theme.

### Packaging

- **Pinned `starlette<1`.** Streamlit 1.60 running on Starlette 1.x answers
  every image, audio and video request with HTTP 500 ("Expected ASGI message
  'http.response.start'"). That blanked the live camera feed and broke
  speech playback and sign videos.

- `pytest` works without `python -m` (`pythonpath` set in `pyproject.toml`).
- `mediapipe==0.10.14` pinned, since newer builds removed `mp.solutions`.
  You get a clear error if the wrong version is installed.
- Python 3.11–3.12 declared (TensorFlow 2.16 has no 3.13 wheels).
  `streamlit>=1.50`.
- `config.yaml` is also found from the project root of an editable install,
  so `sign-bridge` works from any folder.
- Tests: 38 → 52.
