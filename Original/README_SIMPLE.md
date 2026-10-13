# Triage Intelligence — Simple Clinical Design Review

This build replaces only the Streamlit `app.py`. The LangGraph/MCP backend, LightGBM worker, audit database implementation and model training code are unchanged.

## How to run

1. Copy your locally trained `triagegeist_lightgbm_research.joblib` into `models/` (not shipped in this ZIP).
2. From this folder run `python -m pip install -r requirements.txt` if needed.
3. Run `python -m streamlit run app.py`.
4. Enter facility and worker; fill required measurements (including **actually assessed** GCS), then click **Assess & Record**.
5. A fictional record is saved automatically to `demo_register.sqlite3` and displayed in the **Report** tab with its patient number, experimental level and coloured label. A separate backend audit draft is also created. There is no clinical approval in this simple interface; all cases remain **Pending clinical review**. No real queues are touched.

The data and model are synthetic research artefacts, and this app is not for real patient triage. Level colours visually distinguish five model classes; they are **not** a clinically validated SATS mapping.

Note: A local SQLite database may not persist reliably on a free ephemeral hosting instance; use persistent storage before relying on saved demo records across deployments.
