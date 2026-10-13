# Triage Intelligence — isolated-worker MCP version

Research demonstration for fictional patients only. **Not validated for clinical use.**

## Why this version

The preceding project successfully performed direct LightGBM predictions, while
its MCP server stalled inside `core.predict()`. This revision runs that inference
in `agentic/predict_worker.py`, launched as a child process. MCP, LangGraph,
audit handling, and clinician review stay in the main workflow. The five MCP
server tools remain available. This workaround should be verified on the user's
Windows computer; it has not been end-to-end tested in that environment.

## PowerShell (new folder)

1. Extract the ZIP under your `Compitition` directory so that this README and
`demo.py` are in `Triage_Intelligence_Worker_Fix`.
2. Navigate to the new folder:

   cd "C:\Users\E102799\OneDrive - Road Accident Fund\Desktop\Compitition\Triage_Intelligence_Worker_Fix"

3. Copy the model without changing the original:

   Copy-Item "..\Triage_Intelligence_Fresh\models\triagegeist_lightgbm_research.joblib" ".\models\triagegeist_lightgbm_research.joblib"

4. Install packages if necessary:

   python -m pip install -r requirements.txt

5. Run a direct worker smoke test:

   Get-Content .\demo_case.json -Raw | python -u -m agentic.predict_worker

6. Run the complete LangGraph + MCP workflow:

   python -u demo.py

Output should include `[1/3] MCP validation`, `[2/3] MCP LightGBM prediction`,
`[3/3] MCP draft and audit` and a JSON result with `draft.pending_review: true`.
If it does not, examine `Get-Content .\mcp_debug.log -Tail 20`.

The worker never changes queues. Approval is an explicit subsequent action:

   python demo.py --review-case CASE_ID --reviewer "Demo Reviewer" --decision approved_demo_record

Use only fictional test records. No clinical interpretation or automatic queue
placement is authorized.

## Clinical-review Streamlit redesign (13 Oct 2026)

The updated `app.py` is a fictional-patient prototype that connects directly to `agentic.workflow.run` (MCP validation, experimental LightGBM prediction when inputs are complete, and an auditable draft). Human review is saved through the MCP `record_clinician_review` tool. The app shows BP as a grouped systolic/diastolic pair, provides missing-measurement checkboxes (including GCS and SpO2), keeps ML results provisional, and displays reviewed demonstration records separately. The five experimental model levels are **not** validated South African Triage Scale colours.

The `.joblib` trained model is **not included** in this ZIP. Copy it from your existing project into `models/triagegeist_lightgbm_research.joblib` before launching.

PowerShell (inside this project folder):

```powershell
Copy-Item "..\Kenya_Triage_AI_Integration\models\triagegeist_lightgbm_research.joblib" ".\models\triagegeist_lightgbm_research.joblib"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use invented patient data only. Real-world triage, automatic colour decisions, clinical guidance, and real queue changes are expressly unsupported. This version is for review with a qualified clinical advisor.
