"""The 2-persona conversation engine.

Persona 0 (the asker) receives the spec's `topic` as its first message and
speaks first; persona 1 (the responder) replies; they alternate for
`rounds_per_persona` rounds (default 10 → 20 turns). Each persona keeps its
own chat history: its system prompt, then every line the other persona said
(as `[Name]: text` user messages) and its own replies. A persona with
`allow_web_search` gets the `web_search` tool and may call it mid-turn.

Port of the Rust `loop-conversation-gen` engine (conversation.rs) — same
system-prompt suffix, message framing, retry policy and text sanitising.
"""

from __future__ import annotations

import json
import logging
import time

from convgen import web_search
from convgen.langfuse import LangfuseClient, gateway_metadata, session_id
from convgen.llm import LLMClient, ProviderError
from convgen.trace import new_trace, record_turn

log = logging.getLogger("convgen.conversation")

AUTONOMY_NOTE = (
    "\n\nThis is a natural back-and-forth conversation — respond "
    "at conversational length (a few sentences to a short paragraph), not an essay. Stay "
    "fully in character. Never break character to mention you are an AI or a language model. "
    "You are already in the middle of an ongoing conversation — only greet the other person in "
    "your very first message, if at all. Every later message is a direct reply to what was just "
    "said; never re-greet, re-introduce yourself, or restart the conversation as if it were new."
)

ROLES = ("asker", "responder")
MAX_ATTEMPTS = 5          # per turn; backoff 1s, 2s, 4s, 8s between attempts
MAX_TOOL_STEPS = 6        # model requests per turn, including web_search round-trips


def sanitize_text(text: str) -> str:
    """Some models (observed: `soket/gpt` writing non-Latin scripts, e.g. Tamil) emit invisible
    non-standard spaces in place of most regular spaces. Normalise them to a plain space and
    drop the zero-width one so trace text is well-formed whichever model produced it."""
    out = []
    for c in text:
        if c in "   ⁠":
            out.append(" ")
        elif c != "​":
            out.append(c)
    return "".join(out)


def validate_spec(spec: dict) -> None:
    for key in ("id", "topic", "rounds_per_persona", "personas"):
        if key not in spec:
            raise ValueError(f"spec missing {key!r}")
    if len(spec["personas"]) != 2:
        raise ValueError(f"{spec['id']}: this engine runs 2-person conversations, "
                         f"got {len(spec['personas'])} personas")


class Persona:
    def __init__(self, spec: dict, role: str):
        self.name = spec["name"]
        self.model = spec["model"]
        self.role = role
        self.spec = spec
        self.tools = [web_search.TOOL_DEF] if spec.get("allow_web_search") and web_search.enabled() else []
        self.history: list[dict] = [{"role": "system", "content": spec["system_prompt"] + AUTONOMY_NOTE}]


class TurnResult:
    def __init__(self):
        self.text = ""
        self.new_messages: list[dict] = []   # appended to history only if the turn succeeds
        self.steps: list[dict] = []
        self.tool_runs: list[dict] = []
        self.tool_calls: list[dict] = []     # the compact form kept in the local trace
        self.input_tokens = 0
        self.output_tokens = 0


def run_turn(llm: LLMClient, persona: Persona, incoming: str, metadata: dict) -> TurnResult:
    """One attempt at one turn: model request(s) until a text reply, running web_search calls
    in between. Raises ProviderError on failure (the caller retries)."""
    res = TurnResult()
    res.new_messages.append({"role": "user", "content": incoming})
    for _ in range(MAX_TOOL_STEPS):
        messages = persona.history + res.new_messages
        tools = persona.tools if persona.tools else None
        try:
            c = llm.chat(persona.model, messages, tools=tools, metadata=metadata)
        except ProviderError as e:
            res.steps.append({"model": persona.model, "messages": messages, "output": None,
                              "start": time.time(), "end": time.time(), "error": str(e)})
            raise
        res.input_tokens += c.input_tokens
        res.output_tokens += c.output_tokens
        res.steps.append({
            "model": persona.model, "messages": messages, "output": c.message,
            "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
            "start": c.start, "end": c.end, "finish_reason": c.finish_reason, "reasoning": c.reasoning,
        })

        tool_calls = c.message.get("tool_calls") or []
        if tool_calls and persona.tools:
            res.new_messages.append({"role": "assistant", "content": c.message.get("content") or "",
                                     "tool_calls": tool_calls})
            for tc in tool_calls:
                res.new_messages.append(run_tool_call(tc, res))
            continue

        text = sanitize_text(c.message.get("content") or "").strip()
        if not text:
            raise ProviderError(f"empty reply (finish_reason={c.finish_reason!r})")
        res.text = text
        res.new_messages.append({"role": "assistant", "content": text})
        return res
    raise ProviderError(f"no text reply after {MAX_TOOL_STEPS} model requests (tool-call loop)")


def run_tool_call(tc: dict, res: TurnResult) -> dict:
    fn = tc.get("function") or {}
    name = fn.get("name", "")
    start = time.time()
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except ValueError:
        args = {}
    query = str(args.get("query", "")) if isinstance(args, dict) else ""
    is_error = False
    details: dict = {}
    if name != "web_search":
        content, is_error = f"unknown tool {name!r}", True
    elif not query:
        content, is_error = "missing query", True
    else:
        try:
            content, details = web_search.search(query)
        except Exception as e:  # network/HTTP errors go back to the model as the tool result
            content, is_error = f"web_search failed: {e}", True
    res.tool_runs.append({"call_id": tc.get("id"), "name": name, "args": args,
                          "result": details or content, "is_error": is_error,
                          "start": start, "end": time.time()})
    res.tool_calls.append({"tool": name, "query": query, "result_preview": content[:400]})
    return {"role": "tool", "tool_call_id": tc.get("id"), "content": content}


def run_conversation(spec: dict, llm: LLMClient, langfuse: LangfuseClient | None,
                     run_id: str, tags: list[str] | None = None) -> dict:
    validate_spec(spec)
    trace = new_trace(spec)
    session = session_id(spec["id"], run_id)
    started = time.time()
    personas = [Persona(p, ROLES[i]) for i, p in enumerate(spec["personas"])]

    last_line = spec["topic"]
    last_speaker: str | None = None
    for round_ in range(1, int(spec["rounds_per_persona"]) + 1):
        for persona in personas:
            incoming = last_line if last_speaker is None else f"[{last_speaker}]: {last_line}"
            trace_name = f"turn-{round_:02d}-{persona.role}"
            metadata = gateway_metadata(session, trace_name, persona.role, run_id)
            turn_start = time.time()
            result, error = None, ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    result = run_turn(llm, persona, incoming, metadata)
                    break
                except ProviderError as e:
                    error = str(e)
                    log.warning("%s %s: attempt %d/%d failed: %s",
                                spec["id"], persona.name, attempt, MAX_ATTEMPTS, error)
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(1 << (attempt - 1))

            if result is None:
                trace["status"] = "error"
                trace["error"] = f"{persona.name}: provider error after {MAX_ATTEMPTS} attempts: {error}"
                if langfuse:
                    langfuse.turn(session=session, conversation_id=spec["id"], round_=round_,
                                  role=persona.role, persona=persona.spec, incoming=incoming, reply="",
                                  steps=[], tool_runs=[], start=turn_start, end=time.time(),
                                  error=trace["error"], tags=tags)
                    langfuse.conversation(session=session, trace=trace, start=started, tags=tags)
                return trace

            persona.history.extend(result.new_messages)
            record_turn(trace, round_=round_, persona=persona.name, text=result.text,
                        tool_calls=result.tool_calls, input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens)
            if langfuse:
                langfuse.turn(session=session, conversation_id=spec["id"], round_=round_,
                              role=persona.role, persona=persona.spec, incoming=incoming,
                              reply=result.text, steps=result.steps, tool_runs=result.tool_runs,
                              start=turn_start, end=time.time(), tags=tags)
            last_line = result.text
            last_speaker = persona.name

    trace["status"] = "completed"
    if langfuse:
        langfuse.conversation(session=session, trace=trace, start=started, tags=tags)
    return trace
