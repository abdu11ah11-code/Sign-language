# Building the 3D person

These scripts build `src/sign_bridge/ui/static/person.glb` (the person in the
"How to sign" box) and `src/sign_bridge/ui/avatar/person_face.json`. They are
kept so the model can be checked and rebuilt; the app itself only needs the two
output files. `build_all.sh` runs every step in order.

They were written with an AI assistant (Claude, Anthropic), with the course
instructor's permission, and run on 22 Nov 2026 with Blender 5.3 used as a
Python module (`pip install bpy==5.3.0`, Python 3.13). Paths at the top of each
script are the ones used then; change them to rebuild elsewhere.

| Step | Script | What it does |
|---|---|---|
| 1 | `build.py` | MakeHuman body (young adult man), the `default` skeleton (163 bones incl. fingers, jaw, eyes), high-poly eyes — through MPFB 2.0.18 |
| 2 | `cornea.py` | removes the eyes' cornea shell |
| 3 | `ortho.py` | front orthographic render of the head |
| 4 | `fm.py` | MediaPipe Face Mesh landmarks on that render and on the portrait |
| 5 | `warp.py` | warps the portrait onto the head's front view (thin-plate spline on the landmarks), takes the portrait's soft lighting out, makes the blend mask, samples the skin tone |
| 6 | `stage3.py` | freezes the body shape; splits off teeth, tongue and lashes; drops helper geometry |
| 7 | `stage4.py` | finds the 468 face points on the head (rays through the landmarks); finds inside faces (mouth, nostrils) by ray casting; unwraps the head; bakes the head texture (portrait where the face looks forward, skin elsewhere, dark scalp, stubble under the jaw) |
| 8 | `eye_tex.py`, `stage5.py` | eye and teeth materials; drops the lash bands and tongue |
| 9 | `hair.py` | grows 3200 hair cards over the scalp, combed back along the head's surface, with a drawn strand texture; fixed to the head bone |
| 10 | `clothes.py` | shirt, trousers and shoes as shells of the body (same skin weights, so they move with it); hides the body under them |
| 11 | `web_textures.py`, `export.py`, `face_points.py` | small textures, the `.glb`, the face points file |

`render.py` makes quick Cycles previews of any stage (`render.py -- stage7.blend out front:head,angle:body`).

## Sources and licences

* MakeHuman base mesh, targets, skeleton, weights, eyes, fingernail mask: CC0 1.0
  (MPFB `LICENSE.ASSETS.md`; MakeHuman `makehuman/data/eyes/*.mhmat`).
  MPFB's code is GPL-3.0; it is only run here, inside Blender, and is not part of SignBridge.
* `portrait_ref11.jpg`: made by the project author (Abdullah) with Google Gemini;
  AI-generated, not a real person.
* three.js `GLTFLoader` (in `src/sign_bridge/ui/vendor/`): MIT, three.js r160.
