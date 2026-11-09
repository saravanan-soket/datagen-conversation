"""Local trace JSON — one file per conversation, same schema as the original
`artifacts/multilingual_dataset/traces/**.json` files:

    {conversation_id, topic, created_at, personas, transcript, status, error, usage_totals}

Also used to resume: a conversation whose trace already says
`status: completed` is skipped on the next run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def new_trace(spec: dict) -> dict:
    return {
        "conversation_id": spec["id"],
        "topic": spec["topic"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "personas": [
            {"name": p["name"], "model": p["model"], "allow_web_search": bool(p.get("allow_web_search", False))}
            for p in spec["personas"]
        ],
        "transcript": [],
        "status": "running",
        "error": None,
        "usage_totals": {},
    }


def record_turn(trace: dict, *, round_: int, persona: str, text: str, tool_calls: list[dict],
                input_tokens: int, output_tokens: int) -> None:
    totals = trace["usage_totals"].setdefault(persona, {"input": 0, "output": 0})
    totals["input"] += input_tokens
    totals["output"] += output_tokens
    trace["transcript"].append({
        "round": round_,
        "persona": persona,
        "text": text,
        "tool_calls": tool_calls,
        "usage": {"input": input_tokens, "output": output_tokens},
    })


def trace_path(out_dir: Path, conversation_id: str) -> Path:
    """`<language>__<domain>__<index>` ids land in `<out>/<language>/<domain>/`, the layout
    of the original multilingual dataset; anything else goes straight into `<out>/`."""
    parts = conversation_id.split("__")
    if len(parts) == 3:
        return out_dir / parts[0] / parts[1] / f"{conversation_id}.json"
    return out_dir / f"{conversation_id}.json"


def write_trace(out_dir: Path, trace: dict) -> Path:
    path = trace_path(out_dir, trace["conversation_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def already_completed(out_dir: Path, conversation_id: str) -> bool:
    try:
        data = json.loads(trace_path(out_dir, conversation_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("status") == "completed"
