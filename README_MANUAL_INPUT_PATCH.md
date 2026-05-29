# Manual input patch for OpenMetalAM-AI

Replace your root `streamlit_app.py`, `requirements.txt`, and `runtime.txt` with these files.

New page added:

`4 Manual training builder`

It lets users:
- add one training row by hand;
- include target property values for training;
- bulk-paste CSV text copied from Excel or paper tables;
- download an empty training CSV template;
- download manually entered rows as CSV;
- train/predict using uploaded CSV + manual rows together.

Streamlit Cloud main file path:

`streamlit_app.py`
