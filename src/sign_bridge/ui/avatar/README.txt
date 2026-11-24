SignBridge 3D person — what it is made of

../static/person.glb   The 3D person shown in the "How to sign" box (glTF 2.0, skinned).
person_face.json       Where each of the figure's 152 face points (sign_bridge.ui.animation
                       FACE_INDICES, in that order) sits on the person at rest: x, y, z in
                       metres (glTF axes) and the bone it moves with ("head" or "jaw").
                       "orient" = positions in that list of the forehead top, chin and the
                       face's two sides.

Made in Blender 5.3 (as a Python module) by the scripts in tools/avatar/ (see the README
there), on 22 Nov 2026, from:

* MakeHuman data, released CC0 1.0 (public domain) by the MakeHuman project:
  the base body mesh, macro shape targets, the "default" skeleton with finger and face
  bones and its skin weights, and the fingernail mask (all from MPFB 2.0.18,
  github.com/makehumancommunity/mpfb2, data/ folder, LICENSE.ASSETS.md), and the
  high-poly eyes with the brown eye texture (github.com/makehumancommunity/makehuman,
  makehuman/data/eyes). MPFB's own code (GPL-3.0) was only run as a tool inside Blender
  to build the body; none of it is part of SignBridge.
* The face texture: the front portrait from a character reference sheet that the
  project author (Abdullah) made with Google Gemini, 21-22 Nov 2026 (tools/avatar/
  portrait_ref11.jpg). AI-generated, not a photograph of a real person. It was warped
  onto the head with MediaPipe Face Mesh landmarks (found on the portrait and on a
  render of the head) and blended into a skin tone sampled from the same portrait.
* Made for SignBridge (by the build scripts): the hair (hair cards grown on the scalp,
  with a drawn strand texture), the shirt, trousers and shoes (shells of the body), the
  head's UV map and baked texture, and the face points.
