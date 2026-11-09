"""Langfuse trace export via the native ingestion API.

`POST {LANGFUSE_BASE_URL}/api/public/ingestion`, HTTP Basic auth
`LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY`, batched `{id, type, timestamp, body}`
events — Langfuse's own wire format, no SDK needed.

Layout in Langfuse, one **session per conversation** (`<conversation_id>-<run_id>`):
- one `conversation` trace (id = the session id): the whole chat as input
  (`[{role, content}]`, rendered as a chat — asker = user, responder = assistant)
  and the outcome (status, turns, usage) as output;
- one trace per persona turn, `turn-<NN>-<asker|responder>`: the incoming line
  in, the reply out, with one `llm-response` generation per model request
  (model, tokens, latency), a `thinking` span when the model returned
  reasoning, and a `tool:web_search` span per search.

Every request to the Soket gateway also carries `metadata` (`conversation_id`,
`trace_name`, `tags`) so the gateway's own Langfuse logger files its per-call
traces under the same session instead of a random one.

Sends never raise: Langfuse being slow or down logs a warning and generation
carries on.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone

import requests

log = logging.getLogger("convgen.langfuse")


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def session_id(conversation_id: str, run_id: str) -> str:
    return f"{conversation_id}-{run_id}"


def gateway_metadata(session: str, trace_name: str, role: str, run_id: str) -> dict:
    """Request metadata for the Soket gateway's own Langfuse logger. Checked live (in the
    goldloan generator): `conversation_id` picks the session, `trace_name` and `tags` are
    honoured, `trace_id` is ignored."""
    return {
        "conversation_id": session,
        "session_id": session,
        "trace_name": trace_name,
        "tags": ["conversation-gen", "gateway", role, f"run:{run_id}"],
    }


class LangfuseClient:
    def __init__(self, public_key: str, secret_key: str, base_url: str, run_id: str):
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.auth = f"Basic {token}"
        self.url = f"{base_url.rstrip('/')}/api/public/ingestion"
        self.run_id = run_id
        self.http = requests.Session()
        self.failed_batches = 0

    @classmethod
    def from_env(cls, run_id: str) -> "LangfuseClient":
        missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
                   if not os.environ.get(k, "").strip()]
        if missing:
            raise SystemExit(f"Langfuse enabled but {', '.join(missing)} not set "
                             f"(put them in .env, or pass --no-langfuse)")
        return cls(os.environ["LANGFUSE_PUBLIC_KEY"].strip(), os.environ["LANGFUSE_SECRET_KEY"].strip(),
                   os.environ["LANGFUSE_BASE_URL"].strip(), run_id)

    # ---- transport ---------------------------------------------------------

    def send(self, what: str, events: list[tuple[str, dict]], attempts: int = 4) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        batch = {"batch": [{"id": str(uuid.uuid4()), "type": t, "timestamp": now, "body": b}
                           for t, b in events]}
        last = ""
        for attempt in range(1, attempts + 1):
            try:
                r = self.http.post(self.url, data=json.dumps(batch, ensure_ascii=False).encode(),
                                   headers={"Authorization": self.auth, "Content-Type": "application/json"},
                                   timeout=(8, 20))
                # The ingestion endpoint answers 207 even when individual events failed:
                # trust the body's `errors`, not the status code.
                if r.status_code < 300:
                    errors = (r.json() or {}).get("errors") or [] if r.text else []
                    if not errors:
                        return True
                    last = f"HTTP {r.status_code} errors={errors}"
                    if all(400 <= int(e.get("status", 500)) < 500 for e in errors):
                        break  # rejected, retrying won't help
                elif 400 <= r.status_code < 500 and r.status_code != 429:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    break
                else:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
            except (requests.RequestException, ValueError) as e:
                last = str(e)
            if attempt < attempts:
                time.sleep(2 * attempt + random.random())
        self.failed_batches += 1
        log.warning("langfuse %s: giving up: %s", what, last)
        return False

    def tags(self, *extra: str) -> list[str]:
        return ["conversation-gen", f"run:{self.run_id}", *[t for t in extra if t]]

    # ---- traces ------------------------------------------------------------

    def turn(self, *, session: str, conversation_id: str, round_: int, role: str, persona: dict,
             incoming: str, reply: str, steps: list[dict], tool_runs: list[dict],
             start: float, end: float, error: str | None = None, trace_id: str | None = None,
             tags: list[str] | None = None) -> bool:
        """One persona turn. `steps` = one dict per model request:
        {messages, output, model, input_tokens, output_tokens, start, end, finish_reason, reasoning}.
        `tool_runs` = {call_id, name, args, result, is_error, start, end}."""
        trace_id = trace_id or str(uuid.uuid4())
        name = f"turn-{round_:02d}-{role}"
        events: list[tuple[str, dict]] = [("trace-create", {
            "id": trace_id,
            "name": name,
            "sessionId": session,
            "timestamp": iso(start),
            "input": incoming,
            "output": reply,
            "metadata": {
                "conversation_id": conversation_id, "round": round_, "role": role,
                "persona": persona.get("name"), "model": persona.get("model"),
                "allow_web_search": persona.get("allow_web_search", False),
                "model_requests": len(steps), "tool_runs": len(tool_runs), "run_id": self.run_id,
                **({"error": error} if error else {}),
            },
            "tags": self.tags(role, *(tags or [])),
        })]
        n = len(steps)
        for i, st in enumerate(steps):
            g = {
                "id": str(uuid.uuid4()), "traceId": trace_id,
                "name": f"llm-response {i + 1}/{n}" if n > 1 else "llm-response",
                "model": st.get("model"),
                "startTime": iso(st["start"]), "endTime": iso(st["end"]),
                "input": st.get("messages"), "output": st.get("output"),
                "metadata": {"finish_reason": st.get("finish_reason")},
                "usage": {"input": st.get("input_tokens", 0), "output": st.get("output_tokens", 0),
                          "total": st.get("input_tokens", 0) + st.get("output_tokens", 0), "unit": "TOKENS"},
            }
            if st.get("error"):
                g["level"] = "ERROR"
                g["statusMessage"] = st["error"]
            events.append(("generation-create", g))
            if (st.get("reasoning") or "").strip():
                events.append(("span-create", {
                    "id": str(uuid.uuid4()), "traceId": trace_id,
                    "name": f"thinking {i + 1}/{n}" if n > 1 else "thinking",
                    "startTime": iso(st["start"]), "endTime": iso(st["end"]), "output": st["reasoning"],
                }))
        for r in tool_runs:
            sp = {
                "id": str(uuid.uuid4()), "traceId": trace_id, "name": f"tool:{r['name']}",
                "startTime": iso(r["start"]), "endTime": iso(r["end"]),
                "input": r.get("args"), "output": r.get("result"),
                "metadata": {"tool_call_id": r.get("call_id")},
            }
            if r.get("is_error"):
                sp["level"] = "ERROR"
            events.append(("span-create", sp))
        return self.send(f"{name} ({session})", events)

    def conversation(self, *, session: str, trace: dict, start: float, tags: list[str] | None = None) -> bool:
        """The dataset row: the whole chat in, the outcome out. Timestamped at the conversation's
        start so it sorts first in the session; its id is the session id, so re-sending it
        updates the same trace instead of duplicating it."""
        personas = trace["personas"]
        asker = personas[0]["name"] if personas else None
        chat = [{"role": "system", "content": trace["topic"]}] + [
            {"role": "user" if t["persona"] == asker else "assistant",
             "content": t["text"], "name": t["persona"]}
            for t in trace["transcript"]
        ]
        output = {
            "status": trace["status"], "error": trace.get("error"),
            "turns": len(trace["transcript"]), "usage_totals": trace.get("usage_totals", {}),
        }
        metadata = {
            "conversation_id": trace["conversation_id"], "topic": trace["topic"],
            "personas": personas, "created_at": trace.get("created_at"), "run_id": self.run_id,
            "web_search_calls": sum(len(t.get("tool_calls", [])) for t in trace["transcript"]),
        }
        return self.send(f"conversation ({session})", [("trace-create", {
            "id": session, "name": "conversation", "sessionId": session, "timestamp": iso(start),
            "input": chat, "output": output, "metadata": metadata,
            "tags": self.tags("conversation", trace["status"], *(tags or [])),
        })])
