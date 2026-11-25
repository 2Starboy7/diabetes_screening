"""Streamlit Cloud entry point (default main-file name).

Delegates to the real app so either `app.py` or `streamlit_app.py` works as the
deployed main file path.
"""

from pathlib import Path

import runpy

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
