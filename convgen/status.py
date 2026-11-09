"""status.py — tally local traces by status, language and domain.

    python -m convgen.status --traces data/traces
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from convgen.env import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarise generated traces")
    p.add_argument("--traces", type=Path, default=REPO_ROOT / "data" / "traces")
    args = p.parse_args(argv)

    status, combos, turns = Counter(), Counter(), 0
    for f in args.traces.rglob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status["unreadable"] += 1
            continue
        status[d.get("status", "?")] += 1
        turns += len(d.get("transcript", []))
        parts = d.get("conversation_id", "").split("__")
        combos[f"{parts[0]}/{parts[1]}" if len(parts) == 3 else "(other)"] += 1

    total = sum(status.values())
    print(f"{total:,} traces, {turns:,} turns — " + ", ".join(f"{k}: {v:,}" for k, v in status.most_common()))
    for combo, n in sorted(combos.items()):
        print(f"  {combo:40s} {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
