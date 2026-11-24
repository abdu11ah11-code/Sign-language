"""Render the head from the front with an orthographic camera; save the camera frame."""
import bpy, sys, json
from mathutils import Vector
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0]); out = argv[1]
sc = bpy.context.scene
h = bpy.data.objects["Human"]; eyes = bpy.data.objects["Human.high-poly"]
skin = bpy.data.materials.new("skin"); skin.use_nodes = True
skin.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.22, 0.16, 1)
h.data.materials.clear(); h.data.materials.append(skin)
em = bpy.data.materials.new("eye"); em.use_nodes = True
nt = em.node_tree; tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load("/home/claude/dl/mh/makehuman/data/eyes/materials/brown_eye.png")
nt.links.new(tex.outputs[0], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.15
eyes.data.materials.clear(); eyes.data.materials.append(em)
sc.render.engine = "CYCLES"; sc.cycles.samples = 32; sc.cycles.use_denoising = True
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x = sc.render.resolution_y = 1024
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
rig = bpy.data.objects["Human.rig"]
head = rig.matrix_world @ rig.pose.bones["head"].head
centre = head + Vector((0, 0, 0.035)); size = 0.30
cd = bpy.data.cameras.new("c"); cd.type = "ORTHO"; cd.ortho_scale = size
cam = bpy.data.objects.new("c", cd); sc.collection.objects.link(cam); sc.camera = cam
cam.location = centre + Vector((0, -1.0, 0)); cam.rotation_euler = (1.5707963, 0, 0)
sc.render.filepath = out + ".png"; bpy.ops.render.render(write_still=True)
json.dump({"centre": list(centre), "size": size}, open(out + ".json", "w"))
bpy.ops.wm.save_as_mainfile(filepath=argv[0].replace(".blend", "_m.blend"))
print("DONE")
