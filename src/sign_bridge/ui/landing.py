"""The SignBridge landing page, in two forms built from one template (``landing.html``).

* ``"web"``: the public page (``docs/index.html``, served by GitHub Pages). It loads three.js
  from a CDN and the 3D person from ``person.glb`` beside it, and has an **Open the app**
  button that points at the app on the same laptop (``http://localhost:8501``).
* ``"app"``: the welcome screen inside the app. It works offline: three.js and its glTF
  loader are the app's own bundled copies, the 3D person comes from the app's static
  folder, the face points are written into the page, and the Open-the-app buttons become
  **Start signing** buttons that press the app's own (hidden) Start signing button.

Run ``python -m sign_bridge.ui.landing docs/index.html`` to rebuild the public page after
editing the template.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Literal

from sign_bridge.ui.figure3d import _GLTF_LOADER, _VENDOR, PERSON_URL

TEMPLATE = Path(__file__).with_name("landing.html")
FACE_POINTS = Path(__file__).with_name("avatar") / "person_face.json"
APP_URL = "http://localhost:8501"

Mode = Literal["web", "app"]

_IMPORT_MAP = re.compile(r'<script type="importmap">.*?</script>\s*', re.S)
_IMPORTS = ('import * as THREE from "three";\n'
            'import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";\n')
_APP_LINK = re.compile(r"<!--app-link-->.*?<!--/app-link-->", re.S)
_POSTER = re.compile(r'\s*<img src="hello\.jpg"[^>]*>')
# inside the app, the page's own buttons press the app's (hidden) Start signing button
_START = ('<p class="open"><a class="btn" href="#" data-start>Start signing '
          '<span aria-hidden="true">→</span></a>'
          '<span class="label">Opens the camera and the signing tabs</span></p>')
_START_CTA = '<a href="#" data-start>Start signing <em>camera</em></a>'
_START_JS = """
document.querySelectorAll("[data-start]").forEach(a => a.addEventListener("click", e => {
  e.preventDefault();
  try { parent.document.querySelector(".st-key-start_signing button").click(); }
  catch (err) { a.textContent = "Use Start signing at the top of the app"; }
}));
"""


def _split(page: str) -> tuple[str, str, str]:
    """The template's head part (title, metas, font links), its style rules and its body."""
    head, rest = page.split("<style>", 1)
    style, body = rest.split("</style>", 1)
    return head.strip(), style, body.strip()


def _document(head: str, style: str, body: str, theme: str | None = None) -> str:
    root = f' data-theme="{theme}"' if theme else ""
    viewport = "width=device-width, initial-scale=1, viewport-fit=cover"
    return (f'<!doctype html>\n<html lang="en"{root}>\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="{viewport}">\n'
            f"{head}\n<style>{style}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def landing_page(mode: Mode = "web") -> str:
    """The landing page as one complete HTML document.

    Raises:
        ValueError: An unknown mode, or a template that no longer has the parts the app
            form replaces (so a template edit cannot silently break the in-app page).
    """
    page = TEMPLATE.read_text(encoding="utf-8")
    if mode == "web":
        head, style, body = _split(page)
        return _document(head, style, body)
    if mode != "app":
        raise ValueError(f"unknown landing page mode: {mode!r}")
    for part in (_IMPORTS, 'fetch("person.glb")', 'fetch("person_face.json").then(r => r.json())'):
        if part not in page:
            raise ValueError(f"landing.html no longer contains {part!r}")
    # links to the app make no sense inside it; the first becomes a pointer to Start signing
    page = _APP_LINK.sub(_START, page, count=1)
    page = _APP_LINK.sub(_START_CTA, page, count=1)
    page = _APP_LINK.sub("", page)
    page = page.replace('<a class="go" href="#watch">Watch the demo →</a>',
                        '<a class="go" href="#" data-start>Start signing →</a>')
    page = _POSTER.sub("", page)
    page = _IMPORT_MAP.sub("", page)
    page = page.replace(_IMPORTS, "const GLTFLoader = THREE.GLTFLoader;\n")
    page = page.replace('fetch("person.glb")', f'fetch("{PERSON_URL}")')
    face = FACE_POINTS.read_text(encoding="utf-8")
    page = page.replace('fetch("person_face.json").then(r => r.json())',
                        f"Promise.resolve({json.dumps(json.loads(face))})")
    vendor = (f"<script>\n{_VENDOR.read_text(encoding='utf-8')}\n</script>\n"
              f"<script>\n{_GLTF_LOADER.read_text(encoding='utf-8')}\n</script>\n")
    head, style, body = _split(page)
    body = body.replace('<script type="module">', vendor + '<script type="module">', 1)
    body += "\n<script>" + _START_JS + "</script>"
    return _document(head, style, body, theme="light")


def main(argv: list[str] | None = None) -> None:
    """Write the public page: ``python -m sign_bridge.ui.landing docs/index.html``."""
    args = sys.argv[1:] if argv is None else argv
    out = Path(args[0] if args else "docs/index.html")
    out.write_text(landing_page("web"), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
