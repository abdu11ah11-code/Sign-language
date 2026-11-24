#!/usr/bin/env bash
# Rebuild ui/static/person.glb and ui/avatar/person_face.json from scratch.
# B = Blender as a Python module (pip install bpy==5.3.0, Python 3.13);
# P = the project's own venv (MediaPipe 0.10.14, Pillow, OpenCV, SciPy).
# Paths are the ones this was run with (work folder /home/claude/avatar, MPFB 2.0.18 in
# /home/claude/dl/mpfb2, MakeHuman repo in /home/claude/dl/mh); change them to rebuild elsewhere.
set -euo pipefail
B=${B:-/home/claude/bl/bin/python}; P=${P:-/home/claude/venv312/bin/python}
W=/home/claude/avatar; T=$(cd "$(dirname "$0")" && pwd)
mkdir -p $W && cp $T/*.py $T/portrait_ref11.jpg $W/ && cd $W
$B build.py                                    # MakeHuman human + default rig + eyes -> stage1.blend
$B cornea.py -- stage1.blend stage2.blend      # drop the cornea shell
$B ortho.py -- stage2.blend $W/front           # front view for the landmark fit
$P fm.py front.png front_lm.json && $P fm.py portrait_ref11.jpg portrait_lm.json
$P warp.py                                     # portrait warped onto the front view + blend mask
$B stage3.py -- stage2.blend stage3.blend      # freeze shape, split teeth/tongue/lashes
$B stage4.py -- stage3.blend stage4.blend      # head UVs, baked head texture, face points
$P eye_tex.py
$B stage5.py -- stage4.blend stage5.blend      # eyes, teeth
$B hair.py -- stage5.blend stage6.blend 3200   # hair cards
$B clothes.py -- stage6.blend stage7.blend     # shirt, trousers, shoes
$P web_textures.py
$B export.py -- stage7.blend $W/person.glb     # -> person.glb + avatar_face.json
$P face_points.py
cp person.glb $T/../../src/sign_bridge/ui/static/person.glb
