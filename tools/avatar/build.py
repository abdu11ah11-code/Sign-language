"""Stage 1: the human, rig and eyes from MakeHuman CC0 data (Blender Python)."""
import sys, os, math
sys.path.insert(0, "/home/claude/avatar")
from common import *
G = load_mpfb()
HS, TS, HOP = G["HumanService"], G["TargetService"], G["HumanObjectProperties"]
bpy.ops.wm.read_factory_settings(use_empty=True)
h = HS.create_human()
for k, v in {"gender": 1.0, "age": 0.5, "muscle": 0.6, "weight": 0.5, "height": 0.55,
             "african": 0.25, "asian": 0.1, "caucasian": 0.65, "proportions": 0.6}.items():
    HOP.set_value(k, v, entity_reference=h)
TS.reapply_macro_details(h)
rig = HS.add_builtin_rig(h, "default")
eyes = HS.add_mhclo_asset("/home/claude/dl/mh/makehuman/data/eyes/high-poly/high-poly.mhclo", h,
                          asset_type="Eyes", subdiv_levels=0, material_type="NONE")
print("EYES", eyes.name, len(eyes.data.vertices), [m.name for m in eyes.data.materials], eyes.parent.name if eyes.parent else None,
      [md.type for md in eyes.modifiers])
bpy.ops.wm.save_as_mainfile(filepath="/home/claude/avatar/stage1.blend")
print("OK", [o.name for o in bpy.data.objects], [md.type for md in h.modifiers])
