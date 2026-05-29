# Streamlit Cloud deploy fix

Replace your repository root `requirements.txt` with this one and add `runtime.txt` at the repository root.

Why:
- The app is currently trying to install `streamlit==1.36.0` on Python 3.14.
- That pulls `pillow==10.4.0`, which has no compatible wheel in this environment, so Streamlit tries to compile Pillow from source and fails because zlib headers are missing.
- The new requirements use Python-3.14-compatible package versions. The runtime.txt also asks Streamlit Cloud to use Python 3.11 for extra stability.

Streamlit Cloud main file path:

src/app/streamlit_app.py

or, if you have the root wrapper:

streamlit_app.py
