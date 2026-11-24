import bpy, bmesh, numpy as np, sys
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0])
e = bpy.data.objects["Human.high-poly"]; me = e.data
bm = bmesh.new(); bm.from_mesh(me); uvl = bm.loops.layers.uv[0]
dead = [f for f in bm.faces if (lambda c: c[0] > 0.8 and c[1] < 0.2)(sum((l[uvl].uv for l in f.loops), start=__import__("mathutils").Vector((0,0))) / len(f.loops))]
bmesh.ops.delete(bm, geom=dead, context="FACES"); bm.to_mesh(me); bm.free()
bpy.ops.wm.save_as_mainfile(filepath=argv[1]); print("CUT", len(dead))
