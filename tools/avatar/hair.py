"""Stage 6: hair cards grown on the scalp, and a strand texture."""
import bpy, bmesh, sys, math, json, random
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
argv = sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.open_mainfile(filepath=argv[0]); OUT = argv[1]
D = "/home/claude/avatar/"
rng = np.random.default_rng(7); random.seed(7)
h = bpy.data.objects["Human"]; me = h.data; M = h.matrix_world
co = np.array([M @ v.co for v in me.vertices])
polys = [list(p.vertices) for p in me.polygons]
bvh = BVHTree.FromPolygons([Vector(c) for c in co], polys)
lm = json.load(open(D + "avatar_lm3d.json"))
HC = np.array([0.0, lm[168][1] + 0.085, lm[168][2] - 0.01])
top_z = max(c[2] for c in co)
scalp = np.zeros(len(co)); gi = h.vertex_groups["scalp"].index
for v in me.vertices:
    for g in v.groups:
        if g.group == gi: scalp[v.index] = g.weight
# ---- roots: area-weighted samples on scalp faces
tri = []
for p in me.polygons:
    if scalp[list(p.vertices)].mean() > 0.6:
        vs = list(p.vertices)
        for k in range(1, len(vs) - 1): tri.append((vs[0], vs[k], vs[k + 1]))
tri = np.array(tri); a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
NR = int(argv[2]) if len(argv) > 2 else 2600
pick = rng.choice(len(tri), NR, p=area / area.sum())
u, v = rng.random(NR), rng.random(NR); flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
roots = a[pick] + (b[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]

def nearest(p):
    loc, n, _, d = bvh.find_nearest(Vector(p)); return np.array(loc), np.array(n), d

def comb(p):
    d = p - HC; d /= np.linalg.norm(d)
    up = max(d[2], 0.0); side = abs(d[0]); back = max(d[1], 0.0); front = max(-d[1], 0.0)
    s = 1.0 if p[0] >= 0 else -1.0
    top_dir = np.array([0.25 * s * side, 1.0, 0.25])               # swept back
    side_dir = np.array([0.2 * s, 0.95, -0.6])                     # back, behind the ears
    back_dir = np.array([0.15 * s, 0.25, -1.0])                    # down at the back
    w_top, w_side, w_back = up ** 1.5 + 0.2 * front, side ** 2 * (1 - up), back * (1 - up)
    v = w_top * top_dir / np.linalg.norm(top_dir) + w_side * side_dir / np.linalg.norm(side_dir) + w_back * back_dir / np.linalg.norm(back_dir)
    return v / (np.linalg.norm(v) + 1e-9), up, front

SEG = 9
strands = []
for r in roots:
    _, n0, _ = nearest(r)
    cdir, up, front = comb(r)
    length = (0.045 + 0.075 * up + 0.02 * front + 0.02 * max(0.0, (r - HC)[1] / 0.1)) * (0.8 + 0.4 * rng.random())
    seg = length / SEG
    lift = 0.22 + 0.3 * front * up                                # volume at the front
    d = n0 * lift + cdir * (1 - lift); d /= np.linalg.norm(d)
    p = r.copy(); pts = [p.copy()]; layer = rng.random()
    for k in range(SEG):
        loc, n, dist = nearest(p)
        cd, _, _ = comb(p)
        cd = cd - n * np.dot(cd, n) * (0.85 if k > 1 else 0.3)    # follow the head's surface
        cd = cd + np.array([0, 0, -1.0]) * (0.08 + 0.25 * k / SEG) * (1 - 0.6 * up * front)
        cd /= np.linalg.norm(cd)
        d = d + (cd - d) * 0.45
        d /= np.linalg.norm(d)
        p = p + d * seg
        loc, n, dist = nearest(p)
        inside = np.dot(p - loc, n) < 0
        want = 0.003 + 0.007 * min(1, (k + 1) / 4) * (0.5 + layer) + 0.008 * up * front * (1 - k / SEG)
        if inside or dist < want:
            p = loc + n * want
        elif dist > want + 0.012:                                   # don't fly off the head
            p = loc + n * (want + 0.012)
        pts.append(p.copy())
    strands.append(np.array(pts))

# ---- cards: flat ribbons lying over the head
bm = bmesh.new(); uvl = bm.loops.layers.uv.new("UVMap")
cols = 8                                                         # strand texture has 8 columns
for s in strands:
    col = rng.integers(cols); u0, u1 = col / cols, (col + 1) / cols
    w0 = 0.010 + 0.008 * rng.random()
    left, right = [], []
    for k, p in enumerate(s):
        t = s[min(k + 1, SEG)] - s[max(k - 1, 0)]; t /= np.linalg.norm(t)
        _, n, _ = nearest(p)
        side = np.cross(t, n); side /= (np.linalg.norm(side) + 1e-9)
        w = w0 * (1 - 0.65 * k / SEG)
        left.append(bm.verts.new(p - side * w / 2)); right.append(bm.verts.new(p + side * w / 2))
    for k in range(SEG):
        f = bm.faces.new((left[k], right[k], right[k + 1], left[k + 1]))
        vv = (k / SEG, k / SEG, (k + 1) / SEG, (k + 1) / SEG)
        for loop, uu, vvv in zip(f.loops, (u0, u1, u1, u0), vv):
            loop[uvl].uv = (uu, 1 - vvv)
hm = bpy.data.meshes.new("Hair"); bm.to_mesh(hm); bm.free()
hair = bpy.data.objects.new("Hair", hm); bpy.context.collection.objects.link(hair)
# rigid on the head bone
rig = bpy.data.objects["Human.rig"]
from mathutils import Matrix
hair.parent = rig; hair.parent_type = "BONE"; hair.parent_bone = "head"
bpy.context.view_layer.update()
hair.matrix_world = Matrix.Identity(4)                         # stays where it is, moves with the head
for p in hm.polygons: p.use_smooth = True

# ---- strand texture: RGBA, strands along v, tapering tips
TW, TH = 512, 1024
img = np.zeros((TH, TW, 4), np.float32)
yy = np.linspace(0, 1, TH)[:, None]
for col in range(cols):
    x0, x1 = col * TW // cols, (col + 1) * TW // cols
    for _ in range(26):
        cx = rng.uniform(x0 + 2, x1 - 2); wpx = rng.uniform(0.8, 2.2)
        tip = rng.uniform(0.75, 1.0)
        wave = 1.2 * np.sin(yy * rng.uniform(3, 7) + rng.uniform(0, 6))
        xs = np.arange(x0, x1)[None, :]
        prof = np.clip(1 - np.abs(xs - cx - wave) / wpx, 0, 1)
        alpha = prof * np.clip((tip - yy) / 0.15, 0, 1)
        shade = rng.uniform(0.6, 1.2)
        img[:, x0:x1, 3] = np.maximum(img[:, x0:x1, 3], alpha)
        img[:, x0:x1, :3] = np.maximum(img[:, x0:x1, :3], (alpha * shade)[..., None])
base = np.array([0.018, 0.015, 0.013]); hi = np.array([0.07, 0.055, 0.045])
lum = np.clip(img[..., :3].mean(-1, keepdims=True), 0, 1.2)
img[..., :3] = base + (hi - base) * (lum - 0.6).clip(0) * 1.5
img[..., 3] = np.clip(img[..., 3] * 1.6, 0, 1)
srgb = np.where(img[..., :3] <= 0.0031308, img[..., :3] * 12.92, 1.055 * img[..., :3] ** (1 / 2.4) - 0.055)
out = np.dstack([srgb, img[..., 3:]])
im = bpy.data.images.new("hair_tex", TW, TH, alpha=True); im.pixels[:] = out[::-1].ravel()
im.filepath_raw = D + "hair_tex.png"; im.file_format = "PNG"; im.save()
m = bpy.data.materials.new("Hair"); m.use_nodes = True; nt = m.node_tree
t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(D + "hair_tex.png")
bsdf = nt.nodes["Principled BSDF"]
nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"]); nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
bsdf.inputs["Roughness"].default_value = 0.6
bsdf.inputs["Specular IOR Level"].default_value = 0.25
m.blend_method = "CLIP" if hasattr(m, "blend_method") else None
hm.materials.append(m)
print("HAIR", len(strands), len(hm.polygons))
bpy.ops.wm.save_as_mainfile(filepath=OUT)
