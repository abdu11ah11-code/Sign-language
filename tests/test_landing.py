"""Tests for the landing page (ui/landing.py) and the app's welcome screen."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sign_bridge.ui import landing
from sign_bridge.ui.figure3d import PERSON_URL

ROOT = Path(__file__).resolve().parents[1]


def test_web_page_is_a_whole_document_with_the_app_link() -> None:
    page = landing.landing_page("web")
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert "<title>SignBridge</title>" in page
    assert landing.APP_URL in page                       # Open the app
    assert '<script type="importmap">' in page and 'fetch("person.glb")' in page
    assert "© 2026 Abdullah. All rights reserved." in page


def test_app_page_works_offline() -> None:
    page = landing.landing_page("app")
    assert 'data-theme="light"' in page                  # matches the app's light theme
    assert "importmap" not in page and "cdn.jsdelivr" not in page
    assert "from \"three\"" not in page                  # no module imports left
    assert "const GLTFLoader = THREE.GLTFLoader;" in page and "THREE.GLTFLoader =" in page
    assert f'fetch("{PERSON_URL}")' in page
    assert 'fetch("person_face.json")' not in page and "hello.jpg" not in page
    # the face points are written in, and match the app's file
    m = re.search(r"Promise\.resolve\((\{.*?\})\)\.then", page, re.S)
    face = json.loads(landing.FACE_POINTS.read_text(encoding="utf-8"))
    assert m and json.loads(m.group(1)) == face


def test_app_page_points_to_start_signing_instead_of_itself() -> None:
    page = landing.landing_page("app")
    assert landing.APP_URL not in page
    assert "<!--app-link-->" not in page
    assert page.count("data-start") >= 3                  # header, opening, links at the end
    assert ".st-key-start_signing button" in page


def test_numbers_on_the_page_are_the_measured_ones() -> None:
    page = landing.landing_page("web")
    for number in ("72.26", "93.23", "91.60", "92.64", "68.54", "68.98", "89.5", "4.0", "90.7"):
        assert number in page, number


def test_unknown_mode_and_broken_template_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError):
        landing.landing_page("print")  # type: ignore[arg-type]
    broken = tmp_path / "landing.html"
    text = landing.TEMPLATE.read_text(encoding="utf-8").replace('fetch("person.glb")', "x")
    broken.write_text(text, encoding="utf-8")
    monkeypatch.setattr(landing, "TEMPLATE", broken)
    with pytest.raises(ValueError, match="person.glb"):
        landing.landing_page("app")


def test_docs_page_is_built_from_the_template() -> None:
    docs = ROOT / "docs" / "index.html"
    if not docs.is_file():
        pytest.skip("docs/ not in this checkout")
    assert docs.read_text(encoding="utf-8") == landing.landing_page("web"), (
        "docs/index.html is out of date: run python -m sign_bridge.ui.landing docs/index.html")


def test_app_opens_on_the_welcome_screen() -> None:
    source = (ROOT / "src" / "sign_bridge" / "ui" / "app.py").read_text(encoding="utf-8")
    assert 'setdefault("view", "home")' in source
    assert 'key="start_signing"' in source and '"Home page"' in source
    assert "100vh" in source                              # the welcome page fills the window
    assert 'landing_page("app")' in source
