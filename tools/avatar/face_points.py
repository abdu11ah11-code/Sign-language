"""The person's face points for SignBridge's 152 face indices -> ui/avatar/person_face.json."""
import json, sys
sys.path.insert(0, "/home/claude/sb_src/sign_bridge/src")
from sign_bridge.ui.animation import FACE_INDICES
pts = json.load(open("/home/claude/avatar/avatar_face.json"))["points"]
face = [None if pts[i] is None else [*pts[i]["p"], pts[i]["bone"]] for i in FACE_INDICES]
orient = [FACE_INDICES.index(i) for i in (10, 152, 234, 454)]
json.dump({"face": face, "orient": orient},
          open("/home/claude/sb_src/sign_bridge/src/sign_bridge/ui/avatar/person_face.json", "w"),
          separators=(",", ":"))
