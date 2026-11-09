"""Load `.env` from the repo root into the process environment.

Values already exported in the shell win over the file, so a one-off
`SOKET_API_KEY=... python -m convgen.run ...` still works.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> None:
    path = path or REPO_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, val.strip().strip("'\""))
    # Same fallbacks the original loop-ai Soket provider accepted.
    for alias in ("TENSORSTUDIO_API_KEY", "LOOP_API_KEY", "OPENAI_API_KEY"):
        if "SOKET_API_KEY" not in os.environ and alias in os.environ:
            os.environ["SOKET_API_KEY"] = os.environ[alias]


def require(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise SystemExit(f"{name} is not set (put it in .env — see .env.example)")
    return val
