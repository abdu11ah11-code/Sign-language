"""Stage 8: tidy up and export one .glb, plus the face landmark points."""
import bpy, sys, json
import numpy as np
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0]); OUT = argv[1]
D = "/home/claude/avatar/"
rig = bpy.data.objects["Human.rig"]; h = bpy.data.objects["Human"]
# landmark points -> bone they move with (before renaming, using the body's weights)
pts = json.load(open(D + "avatar_lm3d.json"))
co = np.array([h.matrix_world @ v.co for v in h.data.vertices])
gname = {vg.index: vg.name for vg in h.vertex_groups}
jaw_set = {"jaw"} | {b.name for b in rig.data.bones["jaw"].children_recursive}
def bone_for(p):
    i = int(np.argmin(((co - p) ** 2).sum(1))); v = h.data.vertices[i]
    w = {gname[g.group]: g.weight for g in v.groups}
    jw = sum(x for n, x in w.items() if n in jaw_set)
    return "jaw" if jw > 0.5 else "head"
face = []
for p in pts:
    if p is None: face.append(None); continue
    b = bone_for(np.array(p))
    face.append({"p": [round(p[0], 5), round(p[2], 5), round(-p[1], 5)], "bone": b})   # Blender Z-up -> glTF Y-up
# swap textures for the small web versions
swap = {"head_tex.png": "head_tex.jpg", "body_tex.png": "body_tex.jpg", "eye_tex.png": "eye_tex.jpg", "hair_tex.png": "hair_tex_s.png"}
for im in list(bpy.data.images):
    base = im.filepath.split("/")[-1]
    if base in swap:
        im.filepath = D + swap[base]; im.reload()
# remove helper data
me = h.data
for n in ("facing", "scalp", "stubble"):
    if n in me.attributes: me.attributes.remove(me.attributes[n])
for o in bpy.data.objects:
    if o.type == "MESH" and o.name != "Human":
        for uv in list(o.data.uv_layers):
            if uv.name != "UVMap": o.data.uv_layers.remove(uv)
if "proj" in me.uv_layers: me.uv_layers.remove(me.uv_layers["proj"])
for o in bpy.data.objects:
    if o.type == "MESH":
        for vg in list(o.vertex_groups):
            if vg.name not in rig.data.bones: o.vertex_groups.remove(vg)
# dots in bone names are dropped by three.js' loader: use underscores
for b in rig.data.bones: b.name = b.name.replace(".", "_")
for o in bpy.data.objects: o.name = o.name.replace(".", "_")
for d in face:
    if d: d["bone"] = d["bone"].replace(".", "_")
json.dump({"points": face}, open(D + "avatar_face.json", "w"))
# hair material: alpha clip
hm = bpy.data.materials["Hair"]
if hasattr(hm, "surface_render_method"): hm.surface_render_method = "DITHERED"
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=False, export_skins=True,
                          export_animations=False, export_morph=False, export_apply=False,
                          export_image_format="AUTO", export_yup=True, export_texcoords=True, export_normals=True,
                          export_tangents=False, export_cameras=False, export_lights=False)
import os; print("GLB", os.path.getsize(OUT) // 1024, "KB")
