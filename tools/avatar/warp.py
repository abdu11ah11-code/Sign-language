"""Warp the portrait onto the avatar's front view (landmark thin-plate spline) and make the blend mask."""
import json, numpy as np, cv2
from scipy.interpolate import RBFInterpolator
import mediapipe as mp
P = json.load(open("portrait_lm.json")); A = json.load(open("front_lm.json"))
p = np.array(P["pts"])[:, :2]; a = np.array(A["pts"])[:, :2]
img = cv2.imread("/home/claude/avatar/portrait_ref11.jpg")
W = A["size"][0]
# avatar pixel -> portrait pixel
f = RBFInterpolator(a, p, kernel="thin_plate_spline", smoothing=2.0, degree=1)
step = 4
gy, gx = np.mgrid[0:W:step, 0:W:step]
q = f(np.column_stack([gx.ravel(), gy.ravel()]).astype(float)).reshape(gx.shape + (2,)).astype(np.float32)
q = cv2.resize(q, (W, W), interpolation=cv2.INTER_LINEAR)
warped = cv2.remap(img, q[..., 0], q[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
# face-oval mask in avatar space
conn = mp.solutions.face_mesh_connections.FACEMESH_FACE_OVAL
nxt = dict(conn); start = next(iter(nxt)); order = [start]
while nxt[order[-1]] != start: order.append(nxt[order[-1]])
oval = a[order].copy()
c = oval.mean(0); oval = c + (oval - c) * 0.97          # stay just inside the outline
mask = np.zeros((W, W), np.float32); cv2.fillPoly(mask, [oval.astype(np.int32)], 1.0)
mask = cv2.GaussianBlur(mask, (0, 0), 9)
# keep the photo's own hair at the temples out: fade the upper outer corners of the oval
cx = a[:, 0].mean(); half = (a[:, 0].max() - a[:, 0].min()) / 2; eye_y = a[[33, 263], 1].mean()
yy, xx = np.mgrid[0:W, 0:W].astype(np.float32)
corner = np.clip((np.abs(xx - cx) - 0.62 * half) / (0.2 * half), 0, 1) * np.clip((eye_y - yy) / 40.0, 0, 1)
mask *= 1 - corner
mask *= np.clip((yy - a[10, 1]) / 45.0, 0, 1)            # forehead top: plain skin (hair in the photo)
# take the photo's soft lighting out (the 3D lights add their own); keep the detail
f = warped.astype(np.float32) + 1
lum = f.mean(-1, keepdims=True); big = cv2.GaussianBlur(lum, (0, 0), 40)[..., None]
inside = mask > 0.5; ref = lum[inside].mean()
warped = np.clip(f * (0.35 + 0.65 * ref / big), 0, 255).astype(np.uint8)
# base skin tone: median of both cheeks in the portrait
inner = []
for idx in (50, 280, 101, 330, 205, 425, 147, 376):     # cheek points
    x, y = a[idx].astype(int); inner.append(warped[y-10:y+10, x-10:x+10].reshape(-1, 3))
edge = (mask > 0.9) & (mask < 0.97)                      # just inside the blend edge
edge[: int(a[159, 1])] = False; edge[int(a[61, 1]):] = False
px = np.concatenate(inner).astype(float)
skin = 0.5 * np.median(px, axis=0) + 0.5 * np.median(warped[edge].astype(float), axis=0)
cv2.imwrite("face_warp.png", warped); cv2.imwrite("face_mask.png", (mask * 255).astype(np.uint8))
prev = (warped * mask[..., None] + skin * (1 - mask[..., None])).astype(np.uint8)
cv2.imwrite("face_prev.png", prev)
json.dump({"skin_bgr": skin.tolist()}, open("skin.json", "w")); print("skin BGR", skin)
