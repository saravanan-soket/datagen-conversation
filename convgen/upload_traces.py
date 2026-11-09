"""upload_traces.py — push existing local trace JSON files to Langfuse.

For traces generated with `--no-langfuse`, traces from the original Rust
engine (e.g. `artifacts/multilingual_dataset/traces/`), or batches that
failed to send during a run:

    python -m convgen.upload_traces --traces data/traces

Trace and session ids are derived from the conversation id and `--run-id`
(default `backfill`), so uploading the same files again updates the same
Langfuse traces instead of duplicating them. A local trace only has each
turn's final text and token counts (not the raw model requests), so turn
traces carry the line in/out and usage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from convgen.conversation import ROLES
from convgen.env import load_env
from convgen.langfuse import LangfuseClient, session_id

NAMESPACE = uuid.UUID("5d9c3f0e-6a1b-4c39-9a57-1f0b8a2c4e11")


def parse_ts(s: str | None) -> float:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else time.time()
    except ValueError:
        return time.time()


def upload(lf: LangfuseClient, trace: dict, run_id: str, tags: list[str]) -> bool:
    conv_id = trace["conversation_id"]
    session = session_id(conv_id, run_id)
    start = parse_ts(trace.get("created_at"))
    names = [p["name"] for p in trace["personas"]]
    by_name = {p["name"]: p for p in trace["personas"]}
    ok = True
    prev = None
    for i, t in enumerate(trace["transcript"]):
        incoming = trace["topic"] if prev is None else f"[{prev['persona']}]: {prev['text']}"
        role = ROLES[names.index(t["persona"])] if t["persona"] in names else "unknown"
        ts = start + i  # no per-turn timestamps locally; keep them ordered
        usage = t.get("usage", {})
        step = {"model": by_name.get(t["persona"], {}).get("model"), "messages": incoming,
                "output": t["text"], "input_tokens": usage.get("input", 0),
                "output_tokens": usage.get("output", 0), "start": ts, "end": ts + 1}
        tool_runs = [{"call_id": None, "name": c.get("tool", "web_search"), "args": {"query": c.get("query")},
                      "result": c.get("result_preview"), "start": ts, "end": ts + 1}
                     for c in t.get("tool_calls", [])]
        ok &= lf.turn(session=session, conversation_id=conv_id, round_=t["round"], role=role,
                      persona=by_name.get(t["persona"], {"name": t["persona"]}), incoming=incoming,
                      reply=t["text"], steps=[step], tool_runs=tool_runs, start=ts, end=ts + 1,
                      trace_id=str(uuid.uuid5(NAMESPACE, f"{session}:{i}")), tags=tags)
        prev = t
    ok &= lf.conversation(session=session, trace=trace, start=start - 1, tags=tags)
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Upload local conversation traces to Langfuse")
    p.add_argument("--traces", type=Path, required=True, help="A trace .json file or a directory (recursive)")
    p.add_argument("--run-id", default="backfill", help="Run id used in session ids (default: backfill)")
    p.add_argument("--status", default="completed", help="Only upload traces with this status ('all' for every one)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--env-file", type=Path, default=None)
    args = p.parse_args(argv)

    load_env(args.env_file)
    lf = LangfuseClient.from_env(args.run_id)
    files = [args.traces] if args.traces.is_file() else sorted(args.traces.rglob("*.json"))
    sent = failed = skipped = 0
    for f in files:
        if args.limit is not None and sent + failed >= args.limit:
            break
        try:
            trace = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"  skip {f}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if args.status != "all" and trace.get("status") != args.status:
            skipped += 1
            continue
        if upload(lf, trace, args.run_id, args.tag):
            sent += 1
        else:
            failed += 1
        if (sent + failed) % 50 == 0:
            print(f"  {sent + failed:,} uploaded ({failed} failed)", flush=True)
    print(f"=== {sent} uploaded, {failed} failed, {skipped} skipped ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
