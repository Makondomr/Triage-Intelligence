# Triage Intelligence

## Run
1. Extract the whole ZIP.
2. Copy your existing `triagegeist_lightgbm_research.joblib` into the `models` folder (or run `setup_from_existing.ps1` from PowerShell if its source project folder is available).
3. Install requirements if needed: `python -m pip install -r requirements.txt -r requirements_agent.txt`
4. Run: `python -m streamlit run app.py`

The facility dropdown and input fields use a light theme with dark bold text. The interface has Patient assessment and Report tabs. The existing LangGraph/MCP worker is retained.

Clinical review note: The included trained artifact is not distributed in this archive; model training and evaluation provenance should be shared with clinical collaborators. Colours remain the project's existing five-level mapping pending clinical mapping review.
