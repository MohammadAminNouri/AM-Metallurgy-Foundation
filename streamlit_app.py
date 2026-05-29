"""Root Streamlit entrypoint for Streamlit Cloud.

This file exists so Streamlit Cloud's default main file path `streamlit_app.py`
works. The real app lives in `src/app/streamlit_app.py`.
"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "src" / "app" / "streamlit_app.py"), run_name="__main__")
