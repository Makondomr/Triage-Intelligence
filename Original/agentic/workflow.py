"""LangGraph workflow: reuse ONE MCP process/session for every tool in an assessment."""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]

class State(TypedDict, total=False):
    assessment: dict
    actor: str
    quality: dict
    prediction: dict
    draft: dict
    tool_calls: list


def decode(result):
    if result.isError:
        raise RuntimeError(f"MCP tool error: {result.content}")
    for part in result.content:
        if getattr(part, "type", "") == "text":
            value = json.loads(part.text)
            # FastMCP returning a JSON-encoded string: decode once more as needed
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value
    raise RuntimeError("MCP tool returned no text")


def server_params():
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    return StdioServerParameters(command=sys.executable, args=["-u", "-m", "agentic.mcp_server"], cwd=str(ROOT), env=env)


async def call(session, name, payload, timeout=120):
    response = await asyncio.wait_for(session.call_tool(name, arguments=payload), timeout=timeout)
    return decode(response)


async def run(assessment, actor="demo-worker"):
    # Only one process launch and handshake for all three tools.
    async with stdio_client(server_params()) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await asyncio.wait_for(session.initialize(), timeout=60)

            async def validate(state: State):
                print("[1/3] MCP validation", flush=True)
                q = await call(session, "validate_assessment", {"assessment_json": json.dumps(state["assessment"])})
                return {"quality": q, "tool_calls": state.get("tool_calls", []) + ["validate_assessment"]}

            def route(state: State):
                return "predict" if state["quality"].get("ok_to_predict") else "draft"

            async def predict(state: State):
                print("[2/3] MCP LightGBM prediction", flush=True)
                p = await call(session, "predict_synthetic_acuity", {"assessment_json": json.dumps(state["assessment"])})
                return {"prediction": p, "tool_calls": state["tool_calls"] + ["predict_synthetic_acuity"]}

            async def draft(state: State):
                print("[3/3] MCP draft and audit", flush=True)
                prediction = state.get("prediction", {"ok": False, "reason": "Incomplete assessment: prediction withheld"})
                d = await call(session, "create_assessment_draft", {
                    "assessment_json": json.dumps(state["assessment"]),
                    "prediction_json": json.dumps(prediction),
                    "quality_json": json.dumps(state["quality"]),
                    "actor": state.get("actor", "demo-worker")
                })
                return {"draft": d, "tool_calls": state["tool_calls"] + ["create_assessment_draft"]}

            builder = StateGraph(State)
            builder.add_node("validate", validate)
            builder.add_node("predict", predict)
            builder.add_node("draft", draft)
            builder.set_entry_point("validate")
            builder.add_conditional_edges("validate", route, {"predict": "predict", "draft": "draft"})
            builder.add_edge("predict", "draft")
            builder.add_edge("draft", END)
            graph = builder.compile()
            return await graph.ainvoke({"assessment": assessment, "actor": actor, "tool_calls": []})


async def invoke(name, args):
    async with stdio_client(server_params()) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await asyncio.wait_for(session.initialize(), timeout=60)
            return await call(session, name, args)
