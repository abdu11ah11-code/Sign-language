"""Process entry point for the ``sign-bridge`` console script.

Launches the Streamlit server pointed at :mod:`sign_bridge.ui.app`. Using
Streamlit's CLI bootstrap keeps a single source of truth for the UI while
still letting users start the app with a plain ``sign-bridge`` command.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sign_bridge.logging_conf import configure_logging, get_logger

log = get_logger(__name__)


def run_app() -> None:
    """Start the Streamlit server hosting the SignBridge UI."""
    configure_logging()
    app_path = Path(__file__).parent / "ui" / "app.py"

    try:
        from streamlit.web import cli as stcli
    except Exception as exc:  # noqa: BLE001
        log.error("streamlit_missing", error=str(exc))
        print("Streamlit is required. Install with: pip install streamlit")
        raise SystemExit(1) from exc

    log.info("launching_ui", app=str(app_path))
    sys.argv = [
        "streamlit", "run", str(app_path),
        "--theme.base", "light",
        "--theme.primaryColor", "#1E5A45",   # Field Guide green, not Streamlit red
        "--theme.backgroundColor", "#F7F5F0",
        "--theme.textColor", "#181611",
        # serves ui/static/ (the 3D person, person.glb) at /app/static/
        "--server.enableStaticServing", "true",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run_app()
