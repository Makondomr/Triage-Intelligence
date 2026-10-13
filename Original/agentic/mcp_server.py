"""Triage Intelligence MCP tools for fictional patient cases only.

Model inference runs in an isolated child process so that it never blocks the
MCP server's event loop. This is a research prototype, not a medical device.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from agentic import core

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "mcp_debug.log"
mcp = FastMCP("Triage Intelligence Research Tools")


def log(message):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + message + "\n")


@mcp.tool()
def validate_assessment(assessment_json: str) -> str:
    """Check completeness and plausible ranges for a fictional assessment."""
    return json.dumps(core.validate(json.loads(assessment_json)))


@mcp.tool()
async def predict_synthetic_acuity(assessment_json: str) -> str:
    """Run experimental LightGBM using an isolated worker; never use for real care."""
    log("predict: received")
    assessment = json.loads(assessment_json)
    quality = core.validate(assessment)
    if not quality["ok_to_predict"]:
        log("predict: withheld due to data quality")
        return json.dumps({"ok": False, "error": "Assessment incomplete or invalid", "quality": quality})

    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    log("predict: launching isolated worker")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-m", "agentic.predict_worker",
        cwd=str(ROOT), env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(json.dumps(assessment).encode("utf-8")), timeout=60
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
        log("predict: isolated worker timeout")
        return json.dumps({"ok": False, "error": "Research-model worker timed out", "real_world_queue_action": False})

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-1200:]
        log("predict: worker error " + detail.replace("\n", " "))
        return json.dumps({"ok": False, "error": "Research-model worker failed", "details": detail,
                           "real_world_queue_action": False})
    try:
        result = json.loads(stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        log("predict: non-JSON worker response")
        return json.dumps({"ok": False, "error": "Worker returned invalid JSON", "real_world_queue_action": False})
    log("predict: isolated worker returned")
    return json.dumps(result)


@mcp.tool()
def create_assessment_draft(
    assessment_json: str, prediction_json: str, quality_json: str, actor: str = "demo-worker"
) -> str:
    """Save a fictional case pending explicit human review, with an audit trail."""
    return json.dumps(core.draft(json.loads(assessment_json), json.loads(prediction_json),
                                 json.loads(quality_json), actor))


@mcp.tool()
def record_clinician_review(case_id: str, reviewer: str, decision: str, note: str = "") -> str:
    """Record human review of a fictional case; no real-world queue update."""
    return json.dumps(core.review(case_id, reviewer, decision, note))


@mcp.tool()
def get_case_audit(case_id: str) -> str:
    """Read the audit trail of a fictional case."""
    return json.dumps(core.history(case_id))


if __name__ == "__main__":
    log("server: starting")
    mcp.run(transport="stdio")
