"""Darken the MakeHuman brown-eye texture's iris towards the portrait's hazel (run with the project venv)."""
import numpy as np
from PIL import Image
im = Image.open("/home/claude/dl/mh/makehuman/data/eyes/materials/brown_eye.png").convert("RGBA")
a = np.asarray(im).astype(np.float32) / 255
rgb = a[..., :3]
mx, mn = rgb.max(-1), rgb.min(-1); sat = (mx - mn) / (mx + 1e-6)
iris = np.clip((sat - 0.25) / 0.25, 0, 1)[..., None]           # the saturated iris ring
lum = rgb @ np.array([0.3, 0.59, 0.11])
target = np.stack([lum * 0.95, lum * 0.78, lum * 0.55], -1) * 0.75
a[..., :3] = rgb * (1 - iris) + target * iris
Image.fromarray((a * 255).clip(0, 255).astype(np.uint8)).save("/home/claude/avatar/eye_tex.png")
