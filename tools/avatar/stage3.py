"""Stage 3: freeze the shape, split off teeth/tongue/lashes, drop helpers."""
import bpy, bmesh, sys
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0])
h = bpy.data.objects["Human"]
bpy.context.view_layer.objects.active = h
for o in bpy.context.selected_objects: o.select_set(False)
h.select_set(True)
# freeze the body shape
if h.data.shape_keys:
    h.shape_key_add(name="mix", from_mix=True)
    keys = h.data.shape_keys.key_blocks
    mix = keys["mix"].data; co = [v.co.copy() for v in mix]
    h.shape_key_clear()
    for v, c in zip(h.data.vertices, co): v.co = c
for md in list(h.modifiers):
    if md.type == "MASK": h.modifiers.remove(md)

def group_verts(obj, names):
    idx = {obj.vertex_groups[n].index for n in names}
    return {v.index for v in obj.data.vertices if any(g.group in idx and g.weight > 0.5 for g in v.groups)}

def split(name, groups):
    keep = group_verts(h, groups)
    o = h.copy(); o.data = h.data.copy(); o.name = name; o.data.name = name
    bpy.context.collection.objects.link(o)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index not in keep], context="VERTS")
    bm.to_mesh(o.data); bm.free()
    return o

parts = {"Teeth": ["helper-upper-teeth", "helper-lower-teeth"], "Tongue": ["helper-tongue"],
         "Lashes": ["helper-l-eyelashes-1", "helper-l-eyelashes-2", "helper-r-eyelashes-1", "helper-r-eyelashes-2"]}
made = {k: split(k, g) for k, g in parts.items()}
body = group_verts(h, ["body"])
bm = bmesh.new(); bm.from_mesh(h.data)
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index not in body], context="VERTS")
bm.to_mesh(h.data); bm.free()
for n, o in made.items(): print("PART", n, len(o.data.vertices), len(o.data.polygons))
print("BODY", len(h.data.vertices), len(h.data.polygons))
bpy.ops.wm.save_as_mainfile(filepath=argv[1])
