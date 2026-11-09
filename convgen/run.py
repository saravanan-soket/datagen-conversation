"""run.py — generate conversations from task-bank specs.

    python -m convgen.run --spec data/task_bank/agriculture/hindi.jsonl --limit 5

`--spec` is a `.json` spec, a `.jsonl` shard (one spec per line), or a
directory (searched recursively for both). Each conversation is written to
`<out>/<language>/<domain>/<id>.json` and, unless `--no-langfuse`, exported
to Langfuse as it runs. Conversations whose local trace already says
`completed` are skipped, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from convgen import web_search
from convgen.conversation import run_conversation
from convgen.env import REPO_ROOT, load_env, require
from convgen.langfuse import LangfuseClient
from convgen.llm import LLMClient
from convgen.trace import already_completed, write_trace

DEFAULT_OUT = REPO_ROOT / "data" / "traces"


def spec_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix in (".json", ".jsonl") and p.is_file())
    if not path.is_file():
        raise SystemExit(f"--spec {path}: no such file or directory")
    return [path]


def iter_specs(paths: list[Path]) -> Iterator[dict]:
    for path in paths:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if line.strip():
                        try:
                            yield json.loads(line)
                        except ValueError as e:
                            raise SystemExit(f"{path}:{n}: bad JSON: {e}")
        else:
            yield json.loads(path.read_text(encoding="utf-8"))


def select(specs: Iterator[dict], *, offset: int, limit: int | None,
           worker_id: int, total_workers: int) -> Iterator[dict]:
    """Global index i → this worker when i % total_workers == worker_id; then offset/limit."""
    taken = skipped = 0
    for i, spec in enumerate(specs):
        if i % total_workers != worker_id:
            continue
        if skipped < offset:
            skipped += 1
            continue
        if limit is not None and taken >= limit:
            return
        taken += 1
        yield spec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 2-person synthetic conversations")
    p.add_argument("--spec", type=Path, required=True, help=".json spec, .jsonl shard, or directory of them")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Local trace directory (default data/traces)")
    p.add_argument("--limit", type=int, default=None, help="Run at most N conversations (after sharding/offset)")
    p.add_argument("--offset", type=int, default=0, help="Skip the first N selected specs")
    p.add_argument("--workers", type=int, default=1, help="Conversations to run in parallel")
    p.add_argument("--worker-id", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")),
                   help="This worker's index when sharding across machines/pods (default $JOB_COMPLETION_INDEX or 0)")
    p.add_argument("--total-workers", type=int, default=int(os.environ.get("TOTAL_WORKERS", "1")),
                   help="Number of shards (default $TOTAL_WORKERS or 1)")
    p.add_argument("--run-id", default=None, help="Langfuse run id (default: UTC timestamp)")
    p.add_argument("--tag", action="append", default=[], help="Extra Langfuse tag (repeatable)")
    p.add_argument("--no-langfuse", action="store_true", help="Only write local trace JSON")
    p.add_argument("--rerun", action="store_true", help="Re-run conversations that already completed")
    p.add_argument("--rounds", type=int, default=None, help="Override rounds_per_persona (e.g. 2 for a smoke test)")
    p.add_argument("--env-file", type=Path, default=None, help="Env file (default: <repo>/.env)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    load_env(args.env_file)
    require("SOKET_API_KEY")
    if not 0 <= args.worker_id < args.total_workers:
        raise SystemExit(f"--worker-id {args.worker_id} out of range for --total-workers {args.total_workers}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    llm = LLMClient()
    langfuse = None if args.no_langfuse else LangfuseClient.from_env(run_id)
    if not web_search.enabled():
        logging.warning("TAVILY_API_KEY not set — personas with allow_web_search will run without the tool")

    files = spec_files(args.spec)
    specs = select(iter_specs(files), offset=args.offset, limit=args.limit,
                   worker_id=args.worker_id, total_workers=args.total_workers)
    print(f"run_id={run_id} worker={args.worker_id}/{args.total_workers} "
          f"spec files={len(files)} out={args.out} langfuse={'off' if langfuse is None else 'on'}", flush=True)

    counts = {"completed": 0, "error": 0, "skipped": 0}

    def one(spec: dict) -> tuple[str, str]:
        if not args.rerun and already_completed(args.out, spec["id"]):
            return spec["id"], "skipped"
        if args.rounds:
            spec = {**spec, "rounds_per_persona": args.rounds}
        t0 = time.monotonic()
        try:
            trace = run_conversation(spec, llm, langfuse, run_id, tags=args.tag)
        except Exception as e:  # bad spec etc. — report and keep going with the rest
            print(f"  {spec.get('id', '?')}: ERROR {e}", flush=True)
            return spec.get("id", "?"), "error"
        path = write_trace(args.out, trace)
        print(f"  {spec['id']}: {trace['status']} ({len(trace['transcript'])} turns, "
              f"{time.monotonic() - t0:.0f}s) -> {path}"
              + (f" — {trace['error']}" if trace.get("error") else ""), flush=True)
        return spec["id"], trace["status"]

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = set()
        for spec in specs:
            pending.add(pool.submit(one, spec))
            # keep the queue bounded so a 10k-line shard isn't all submitted at once
            if len(pending) >= args.workers * 4:
                done = next(as_completed(pending))
                pending.remove(done)
                counts[done.result()[1]] += 1
        for fut in as_completed(pending):
            counts[fut.result()[1]] += 1

    total = sum(counts.values())
    print(f"\n=== {counts['completed']} completed, {counts['error']} error, "
          f"{counts['skipped']} already-completed skipped (of {total}) in {time.monotonic() - start:.0f}s ===")
    if langfuse and langfuse.failed_batches:
        print(f"WARNING: {langfuse.failed_batches} Langfuse batch(es) failed to send — "
              f"re-send with: python -m convgen.upload_traces --traces {args.out}")
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
