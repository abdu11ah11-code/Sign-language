"""Stage 5: materials for eyes, teeth, tongue, lashes."""
import bpy, sys
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0])
def lin(c): return [(x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
def flat(name, srgb, rough=0.5):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]; b.inputs["Base Color"].default_value = (*lin(srgb), 1)
    b.inputs["Roughness"].default_value = rough; return m
eyes = bpy.data.objects["Human.high-poly"]; eyes.name = "Eyes"
em = bpy.data.materials.new("Eye"); em.use_nodes = True; nt = em.node_tree
t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load("/home/claude/avatar/eye_tex.png")
nt.links.new(t.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.25
eyes.data.materials.clear(); eyes.data.materials.append(em)
bpy.data.objects.remove(bpy.data.objects["Lashes"])          # thick helper bands; the photo has the lashes
teeth = bpy.data.objects["Teeth"]
for v in teeth.data.vertices: v.co.y += 0.007 / teeth.matrix_world.to_scale()[1]
bpy.data.objects.remove(bpy.data.objects["Tongue"])           # hidden behind the teeth anyway
for name, col, r in (("Teeth", (0.78, 0.75, 0.68), 0.35),):
    o = bpy.data.objects[name]; o.data.materials.clear(); o.data.materials.append(flat(name, col, r))
bpy.ops.wm.save_as_mainfile(filepath=argv[1]); print("SAVED")
