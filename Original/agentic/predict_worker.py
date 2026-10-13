"""Isolated research-model inference worker: one JSON request on stdin, one response on stdout.

The MCP tool launches this as a separate Python process so numerical model inference
cannot block the MCP server's protocol event loop. For fictional assessments only.
"""
import json
import sys
from agentic import core


def main():
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("Assessment must be a JSON object")
        result = core.predict(request)
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        # Never put logging or traceback on stdout: it carries machine-readable JSON.
        print(f"Model worker failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
