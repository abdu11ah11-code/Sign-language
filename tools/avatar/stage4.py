"""Stage 4: head UV + baked head texture, body texture, face landmark points."""
import bpy, bmesh, sys, json, math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0]); OUT = argv[1]
D = "/home/claude/avatar/"
h = bpy.data.objects["Human"]; rig = bpy.data.objects["Human.rig"]; me = h.data
M = h.matrix_world
co = np.array([M @ v.co for v in me.vertices])
nrm = np.array([(M.to_3x3() @ v.normal).normalized() for v in me.vertices])
cam = json.load(open(D + "front.json")); C0 = np.array(cam["centre"]); S = cam["size"]
lm = json.load(open(D + "front_lm.json")); W = lm["size"][0]
skin_bgr = json.load(open(D + "skin.json"))["skin_bgr"]
skin_srgb = [skin_bgr[2] / 255, skin_bgr[1] / 255, skin_bgr[0] / 255]
def lin(c): return [(x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
skin_lin = lin(skin_srgb)
HAIR_lin = lin([0.075, 0.065, 0.06])

# ---- face landmarks on the mesh (rays from the front camera)
bvh = BVHTree.FromPolygons([Vector(c) for c in co], [list(p.vertices) for p in me.polygons])
pts3 = []
for (px, py, _) in lm["pts"]:
    x = C0[0] + (px / W - 0.5) * S; z = C0[2] - (py / W - 0.5) * S
    hit = bvh.ray_cast(Vector((x, C0[1] - 1.0, z)), Vector((0, 1, 0)))
    pts3.append(list(hit[0]) if hit[0] is not None else None)
print("LANDMARK HITS", sum(p is not None for p in pts3), "/", len(pts3))
json.dump(pts3, open(D + "avatar_lm3d.json", "w"))
chin = np.array(pts3[152]); top = np.array(pts3[10])

# ---- which faces belong to the head (bones under 'head')
head_bones = {b.name for b in rig.data.bones["head"].children_recursive} | {"head"}
gidx = {vg.index: vg.name for vg in h.vertex_groups}
headw = np.zeros(len(co))
for v in me.vertices:
    headw[v.index] = sum(g.weight for g in v.groups if gidx[g.group] in head_bones)
def group_w(name):
    gi = h.vertex_groups[name].index; w = np.zeros(len(co))
    for v in me.vertices:
        for g in v.groups:
            if g.group == gi: w[v.index] = g.weight
    return w
scalp = group_w("scalp")
poly_head = np.array([headw[list(p.vertices)].mean() > 0.5 for p in me.polygons])
print("HEAD FACES", poly_head.sum())
HC_ = None

# ---- per-vertex paint weights
facing = np.clip((-nrm[:, 1] - 0.15) / 0.45, 0, 1); facing = facing * facing * (3 - 2 * facing)
# stubble under the jaw: below the chin line, in front of the neck
neck_y = co[headw > 0.5][:, 1].mean()
below = np.clip((chin[2] + 0.012 - co[:, 2]) / 0.02, 0, 1) * np.clip((co[:, 2] - (chin[2] - 0.075)) / 0.03, 0, 1)
front = np.clip((neck_y + 0.01 - co[:, 1]) / 0.03, 0, 1)
stub = below * front * np.clip(1 - np.abs(co[:, 0]) / 0.075, 0, 1)
for name, vals in (("facing", facing), ("scalp", scalp), ("stubble", stub)):
    a = me.attributes.new(name, "FLOAT", "POINT"); a.data.foreach_set("value", vals.astype(np.float32))

# ---- UV maps: front projection (for the photo) and the head's own map
def uv_layer(name, fn):
    lay = me.uv_layers.new(name=name)
    for poly in me.polygons:
        for li in poly.loop_indices:
            lay.data[li].uv = fn(co[me.loops[li].vertex_index])
    return lay
uv_layer("proj", lambda p: ((p[0] - C0[0]) / S + 0.5, (p[2] - C0[2]) / S + 0.5))
eyes_c = np.array(pts3[168])          # between the eyes
HC = np.array([0.0, eyes_c[1] + 0.085, eyes_c[2] - 0.01])
def sph(p):
    d = p - HC; az = math.atan2(d[0], -d[1]); el = math.atan2(d[2], math.hypot(d[0], d[1]))
    return (0.5 + az / (2 * math.pi), 0.17 + 0.83 * (0.5 + el / math.pi * 1.15))
lay = uv_layer("head_uv", sph)
# faces inside the mouth, nostrils and eye sockets point inwards: keep them off the face map
inner = np.zeros(len(me.polygons), bool)
eyes = bpy.data.objects["Human.high-poly"]
ec = [eyes.matrix_world @ v.co for v in eyes.data.vertices]
allv = [Vector(c) for c in co] + ec
allp = [list(p.vertices) for p in me.polygons] + [[i + len(co) for i in p.vertices] for p in eyes.data.polygons]
occ = BVHTree.FromPolygons(allv, allp)
for poly in me.polygons:
    if not poly_head[poly.index]: continue
    c = co[list(poly.vertices)].mean(0); n = np.array(M.to_3x3() @ poly.normal)
    if abs(c[0]) > 0.055 or c[2] < chin[2] + 0.004 or c[2] > top[2] or c[1] > HC[1] - 0.03:
        continue                       # only the mouth, nose and eye area has inside faces
    r = c - HC; r = r / np.linalg.norm(r)
    if float(np.dot(n, r)) < -0.05:
        inner[poly.index] = True; continue
    d = r + 0.3 * n; d = d / np.linalg.norm(d)
    hit = occ.ray_cast(Vector(c + d * 0.0015), Vector(d), 0.25)
    inner[poly.index] = hit[0] is not None
print("INNER FACES", inner.sum())
poly_head &= ~inner
# the head's own UV map: Blender's smart unwrap of just the head faces (no overlaps)
me.uv_layers.active = lay
bpy.context.view_layer.objects.active = h
for o in bpy.context.view_layer.objects: o.select_set(False)
h.select_set(True)
for poly in me.polygons: poly.select = bool(poly_head[poly.index])
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_mode(type="FACE")
bpy.ops.uv.smart_project(angle_limit=math.radians(70), island_margin=0.004, area_weight=0.0, scale_to_bounds=True)
bpy.ops.object.mode_set(mode="OBJECT")
me.uv_layers.active = me.uv_layers["UVMap"]; me.uv_layers["UVMap"].active_render = True

# ---- head material: photo where it faces the camera, skin elsewhere, dark scalp
def img(path, cs="sRGB"):
    im = bpy.data.images.load(path); im.colorspace_settings.name = cs; return im
mh = bpy.data.materials.new("head_bake"); mh.use_nodes = True; nt = mh.node_tree; N = nt.nodes; L = nt.links
for n in list(N): N.remove(n)
out = N.new("ShaderNodeOutputMaterial"); emit = N.new("ShaderNodeEmission")
uvp = N.new("ShaderNodeUVMap"); uvp.uv_map = "proj"
photo = N.new("ShaderNodeTexImage"); photo.image = img(D + "face_warp.png"); photo.extension = "EXTEND"
mask = N.new("ShaderNodeTexImage"); mask.image = img(D + "face_mask.png", "Non-Color"); mask.extension = "CLIP"
L.new(uvp.outputs[0], photo.inputs[0]); L.new(uvp.outputs[0], mask.inputs[0])
def attr(name):
    a = N.new("ShaderNodeAttribute"); a.attribute_name = name; return a.outputs["Fac"]
mul = N.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
L.new(mask.outputs["Color"], mul.inputs[0]); L.new(attr("facing"), mul.inputs[1])
# skin with a little blotchy variation
noise = N.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value = 60.0
tc = N.new("ShaderNodeTexCoord"); L.new(tc.outputs["Object"], noise.inputs["Vector"])
vary = N.new("ShaderNodeMix"); vary.data_type = "RGBA"; vary.blend_type = "MULTIPLY"
vary.inputs["Factor"].default_value = 0.12
vary.inputs[6].default_value = (*skin_lin, 1); L.new(noise.outputs["Color"], vary.inputs[7])
mix_photo = N.new("ShaderNodeMix"); mix_photo.data_type = "RGBA"
L.new(mul.outputs[0], mix_photo.inputs["Factor"]); L.new(vary.outputs[2], mix_photo.inputs[6]); L.new(photo.outputs["Color"], mix_photo.inputs[7])
# stubble under the jaw: speckled dark
sn = N.new("ShaderNodeTexNoise"); sn.inputs["Scale"].default_value = 900.0; L.new(tc.outputs["Object"], sn.inputs["Vector"])
sfac = N.new("ShaderNodeMath"); sfac.operation = "MULTIPLY"; L.new(attr("stubble"), sfac.inputs[0]); L.new(sn.outputs["Fac"], sfac.inputs[1])
sfac2 = N.new("ShaderNodeMath"); sfac2.operation = "MULTIPLY"; L.new(sfac.outputs[0], sfac2.inputs[0]); sfac2.inputs[1].default_value = 1.1
mix_stub = N.new("ShaderNodeMix"); mix_stub.data_type = "RGBA"; mix_stub.clamp_factor = True
L.new(sfac2.outputs[0], mix_stub.inputs["Factor"]); L.new(mix_photo.outputs[2], mix_stub.inputs[6]); mix_stub.inputs[7].default_value = (*HAIR_lin, 1)
# scalp: hair colour (the hair cards sit on top)
sc_r = N.new("ShaderNodeMapRange"); L.new(attr("scalp"), sc_r.inputs["Value"]); sc_r.inputs[1].default_value = 0.2; sc_r.inputs[2].default_value = 0.9
mix_hair = N.new("ShaderNodeMix"); mix_hair.data_type = "RGBA"
L.new(sc_r.outputs[0], mix_hair.inputs["Factor"]); L.new(mix_stub.outputs[2], mix_hair.inputs[6]); mix_hair.inputs[7].default_value = (*HAIR_lin, 1)
L.new(mix_hair.outputs[2], emit.inputs["Color"]); L.new(emit.outputs[0], out.inputs["Surface"])
target = N.new("ShaderNodeTexImage")
head_img = bpy.data.images.new("head_tex", 2048, 2048); target.image = head_img
N.active = target; target.select = True

# bake on a copy that holds only the head faces
hc = h.copy(); hc.data = me.copy(); bpy.context.collection.objects.link(hc)
for md in list(hc.modifiers): hc.modifiers.remove(md)
bm = bmesh.new(); bm.from_mesh(hc.data)
bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[f for f in bm.faces if not poly_head[f.index]], context="FACES")
bm.to_mesh(hc.data); bm.free()
hc.data.materials.clear(); hc.data.materials.append(mh)
hc.data.uv_layers.active = hc.data.uv_layers["head_uv"]
sc = bpy.context.scene; sc.render.engine = "CYCLES"; sc.cycles.samples = 1; sc.cycles.device = "CPU"
sc.render.bake.margin = 16
for o in bpy.context.view_layer.objects: o.select_set(False)
hc.select_set(True); bpy.context.view_layer.objects.active = hc
bpy.ops.object.bake(type="EMIT", uv_layer="head_uv", margin=16)
head_img.filepath_raw = D + "head_tex.png"; head_img.file_format = "PNG"; head_img.save()
bpy.data.objects.remove(hc)
print("BAKED HEAD")

# ---- materials on the real mesh: head (head_uv) and body (UVMap)
def tex_mat(name, image_path, uvname, rough=0.55):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = img(image_path)
    u = nt.nodes.new("ShaderNodeUVMap"); u.uv_map = uvname
    nt.links.new(u.outputs[0], t.inputs[0])
    b = nt.nodes["Principled BSDF"]; nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    return m
# body texture: skin + fingernails (MakeHuman masks are in UVMap space)
import struct
nails = bpy.data.images.load("/home/claude/dl/mpfb2/src/mpfb/data/textures/mpfb_fingernails.jpg")
n = np.array(nails.pixels[:]).reshape(nails.size[1], nails.size[0], 4)[..., 0]
size = 1024; step = nails.size[0] // size
n = n[::step, ::step]
body = np.ones((size, size, 4), np.float32)
nail_col = np.array(lin([min(1, skin_srgb[0] * 1.15), skin_srgb[1] * 1.0, skin_srgb[2] * 0.98]))
for k in range(3): body[..., k] = skin_lin[k] * (1 - n) + nail_col[k] * n
bimg = bpy.data.images.new("body_tex", size, size); bimg.colorspace_settings.name = "Linear Rec.709"
bimg.pixels[:] = body.ravel(); bimg.filepath_raw = D + "body_tex.png"; bimg.file_format = "PNG"
bimg.save()
mat_head = tex_mat("Skin_head", D + "head_tex.png", "head_uv")
mat_body = tex_mat("Skin_body", D + "body_tex.png", "UVMap")
mat_in = bpy.data.materials.new("Mouth"); mat_in.use_nodes = True
mat_in.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*lin([0.32, 0.12, 0.11]), 1)
me.materials.clear(); me.materials.append(mat_body); me.materials.append(mat_head); me.materials.append(mat_in)
for poly in me.polygons: poly.material_index = 2 if inner[poly.index] else (1 if poly_head[poly.index] else 0)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print("SAVED")
