"""Stage 7: a dark long-sleeved shirt with an open collar, dark trousers and shoes."""
import bpy, bmesh, sys, math, json
import numpy as np
from mathutils import Vector
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0]); OUT = argv[1]
D = "/home/claude/avatar/"
h = bpy.data.objects["Human"]; rig = bpy.data.objects["Human.rig"]; me = h.data; M = h.matrix_world
co = np.array([M @ v.co for v in me.vertices])
gname = {vg.index: vg.name for vg in h.vertex_groups}
W = {}
for v in me.vertices:
    for g in v.groups:
        W.setdefault(gname[g.group], np.zeros(len(co)))[v.index] = g.weight
def wsum(prefixes):
    out = np.zeros(len(co))
    for n, w in W.items():
        if n.startswith(prefixes): out += w
    return out
def bone_head(n): return np.array(rig.matrix_world @ rig.data.bones[n].head_local)
def bone_tail(n): return np.array(rig.matrix_world @ rig.data.bones[n].tail_local)
lm = json.load(open(D + "avatar_lm3d.json")); chin = np.array(lm[152])
torso = wsum(("spine", "clavicle", "shoulder01", "breast", "upperarm", "lowerarm"))
neck = wsum(("neck",)); legs = wsum(("upperleg", "lowerleg", "pelvis")); feet = wsum(("foot", "toe"))
hip_z = bone_head("upperleg01.L")[2]
# how far along each forearm a point is (0 elbow .. 1 wrist)
def along(side):
    e, w = bone_head("lowerarm01." + side), bone_head("wrist." + side)
    d = w - e; return ((co - e) @ d) / (d @ d)
fore = np.maximum(np.where(co[:, 0] > 0, along("L"), -9), np.where(co[:, 0] < 0, along("R"), -9))
neck_base = bone_head("neck01")
# open collar: a V at the front of the neck
v_open = (co[:, 1] < neck_base[1] - 0.02) & (np.abs(co[:, 0]) < (co[:, 2] - (neck_base[2] - 0.13)) * 0.55)
shirt_v = (torso + 0.3 * legs > 0.5) & (neck < 0.35) & (fore < 0.86) & (co[:, 2] > hip_z - 0.02)
arms = wsum(("lowerarm", "wrist", "finger", "metacarpal", "upperarm"))
pants_v = (co[:, 2] <= hip_z + 0.06) & (feet < 0.5) & (arms < 0.05)
shoe_v = feet > 0.2
def faces_of(mask): return np.array([mask[list(p.vertices)].all() for p in me.polygons])

def make(name, fmask, push, srgb, rough, thick):
    o = h.copy(); o.data = me.copy(); o.name = name; o.data.name = name
    bpy.context.collection.objects.link(o)
    bm = bmesh.new(); bm.from_mesh(o.data); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not fmask[f.index]], context="FACES")
    bm.normal_update()
    for v in bm.verts: v.co += v.normal * push
    # relax once so the cloth doesn't show every muscle
    for _ in range(2):
        bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    # a clean hem: smooth each open edge along itself (no zig-zag from the body's faces)
    edge = [v for v in bm.verts if v.is_boundary]
    for _ in range(12):
        new = {}
        for v in edge:
            nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            if len(nb) == 2:
                new[v] = v.co * 0.5 + (nb[0].co + nb[1].co) * 0.25
        for v, c in new.items(): v.co = c
    bm.to_mesh(o.data); bm.free()
    for md in list(o.modifiers):
        if md.type != "ARMATURE": o.modifiers.remove(md)
    s = o.modifiers.new("thick", "SOLIDIFY"); s.thickness = thick; s.offset = -1.0; s.use_rim = True
    bpy.context.view_layer.objects.active = o
    for x in bpy.context.view_layer.objects: x.select_set(False)
    o.select_set(True)
    bpy.ops.object.modifier_move_to_index(modifier="thick", index=0)
    bpy.ops.object.modifier_apply(modifier="thick")
    lin = [(x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in srgb]
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (*lin, 1)
    b.inputs["Roughness"].default_value = rough
    o.data.materials.clear(); o.data.materials.append(m)
    for p in o.data.polygons: p.use_smooth = True
    print("CLOTH", name, len(o.data.vertices))
    return o
shirt_f = faces_of(shirt_v); pants_f = faces_of(pants_v); shoe_f = faces_of(shoe_v)
make("Shirt", shirt_f, 0.006, (0.11, 0.115, 0.13), 0.85, 0.002)
make("Trousers", pants_f & ~shirt_f, 0.005, (0.16, 0.17, 0.2), 0.8, 0.002)
make("Shoes", shoe_f, 0.008, (0.05, 0.045, 0.04), 0.4, 0.003)
# hide the body under the clothes (one ring in from the edges so nothing peeks through)
covered = shirt_v | pants_v | shoe_v
inner = covered.copy()
for p in me.polygons:
    vs = list(p.vertices)
    if not covered[vs].all(): inner[vs] = False
bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
dead = [f for f in bm.faces if all(inner[v.index] for v in f.verts)]
bmesh.ops.delete(bm, geom=dead, context="FACES_ONLY"); bm.to_mesh(me); bm.free()
print("BODY FACES LEFT", len(me.polygons))
bpy.ops.wm.save_as_mainfile(filepath=OUT)
