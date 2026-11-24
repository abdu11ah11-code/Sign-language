"""Small web copies of the textures for the .glb (run with the project venv)."""
from PIL import Image
D = "/home/claude/avatar/"
Image.open(D + "head_tex.png").convert("RGB").save(D + "head_tex.jpg", quality=86)
Image.open(D + "body_tex.png").convert("RGB").resize((256, 256)).save(D + "body_tex.jpg", quality=90)
Image.open(D + "eye_tex.png").convert("RGB").resize((512, 512)).save(D + "eye_tex.jpg", quality=88)
Image.open(D + "hair_tex.png").resize((256, 512)).save(D + "hair_tex_s.png", optimize=True)
