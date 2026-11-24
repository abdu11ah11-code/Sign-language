"""render.py BLEND OUT [views] — quick Cycles previews of the avatar."""
import bpy, sys, math
from mathutils import Vector
argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else sys.argv[1:]
blend, out = argv[0], argv[1]
views = argv[2].split(",") if len(argv) > 2 else ["front", "side", "angle"]
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
sc.render.engine = "CYCLES"; sc.cycles.samples = 24; sc.cycles.device = "CPU"
sc.render.resolution_x = sc.render.resolution_y = 520
sc.cycles.use_denoising = True
sc.view_settings.view_transform = "Standard"
world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.8, 0.82, 0.85, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
for o in list(sc.objects):
    if o.type in ("LIGHT", "CAMERA"): bpy.data.objects.remove(o)
def light(name, loc, energy, size=1.5):
    l = bpy.data.lights.new(name, "AREA"); l.energy = energy; l.size = size
    o = bpy.data.objects.new(name, l); sc.collection.objects.link(o); o.location = loc
    d = Vector((0, 0, 1.5)) - Vector(loc); o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
light("key", (1.2, -2.0, 2.2), 130); light("fill", (-1.6, -1.5, 1.6), 50); light("rim", (0.3, 2.0, 2.2), 90)
# head position from the rig
rig = next(o for o in sc.objects if o.type == "ARMATURE")
head = rig.matrix_world @ rig.pose.bones["head"].head
cam_d = bpy.data.cameras.new("c"); cam = bpy.data.objects.new("c", cam_d); sc.collection.objects.link(cam); sc.camera = cam
target = {"head": head + Vector((0, 0, 0.08)), "body": head + Vector((0, 0, -0.35))}
import os
for v in views:
    name, _, zoom = v.partition(":"); zoom = zoom or "head"
    yaw = {"front": 0, "side": 90, "angle": 40, "back": 180, "left": -40}[name]
    dist = 0.75 if zoom == "head" else 2.1
    t = target[zoom]
    cam.location = t + Vector((math.sin(math.radians(yaw)) * dist, -math.cos(math.radians(yaw)) * dist, 0.02))
    cam.rotation_euler = (t - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_d.lens = 70 if zoom == "head" else 50
    sc.render.filepath = f"{out}_{name}_{zoom}.png"
    bpy.ops.render.render(write_still=True)
print("RENDERED")
