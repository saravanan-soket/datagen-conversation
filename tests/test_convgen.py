"""Offline tests — no network, no keys. Run: python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from convgen import conversation, task_bank, trace, web_search
from convgen.domains import DOMAINS
from convgen.languages import LANGUAGES
from convgen.llm import Completion, ProviderError
from convgen.run import iter_specs, select

LANG = {l.slug: l for l in LANGUAGES}
DOM = {d.slug: d for d in DOMAINS}


def spec(rounds: int = 2, search: bool = False) -> dict:
    return {
        "id": "english__banking__00000", "topic": "Opening question?", "rounds_per_persona": rounds,
        "personas": [
            {"name": "Asha", "model": "soket/qwen3-8-27b", "system_prompt": "You ask.", "allow_web_search": False},
            {"name": "Ravi", "model": "soket/gpt", "system_prompt": "You answer.", "allow_web_search": search},
        ],
    }


def reply(content=None, tool_calls=None) -> Completion:
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return Completion(message=msg, finish_reason="stop", input_tokens=10, output_tokens=5, start=0.0, end=1.0)


class FakeLLM:
    """Replays scripted replies; records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, model, messages, tools=None, metadata=None):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages], "tools": tools,
                           "metadata": metadata})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLangfuse:
    def __init__(self):
        self.turns, self.conversations = [], []

    def turn(self, **kw):
        self.turns.append(kw)
        return True

    def conversation(self, **kw):
        self.conversations.append(kw)
        return True


class TaskBankTests(unittest.TestCase):
    def test_only_pair_domains(self):
        self.assertEqual(len(DOMAINS), 20)
        for d in DOMAINS:
            self.assertEqual(len(d.roles), 2, d.slug)

    def test_deterministic_and_two_personas(self):
        a = task_bank.build_spec(LANG["hindi"], DOM["agriculture"], 7)
        b = task_bank.build_spec(LANG["hindi"], DOM["agriculture"], 7)
        self.assertEqual(a, b)
        self.assertEqual(a["id"], "hindi__agriculture__00007")
        self.assertEqual(len(a["personas"]), 2)
        self.assertEqual([p["model"] for p in a["personas"]], task_bank.PAIR_MODELS)
        self.assertEqual(a["rounds_per_persona"], 10)

    def test_every_combo_builds(self):
        for d in DOMAINS:
            for l in LANGUAGES:
                task_bank.build_spec(l, d, 0)


class ConversationTests(unittest.TestCase):
    def test_turn_order_framing_and_history(self):
        llm = FakeLLM([reply(f"line {i}") for i in range(4)])
        lf = FakeLangfuse()
        t = conversation.run_conversation(spec(rounds=2), llm, lf, "run1")
        self.assertEqual(t["status"], "completed")
        self.assertEqual([x["persona"] for x in t["transcript"]], ["Asha", "Ravi", "Asha", "Ravi"])
        # asker gets the raw topic first, then "[Name]: text" framing
        self.assertEqual(llm.calls[0]["messages"][-1], {"role": "user", "content": "Opening question?"})
        self.assertEqual(llm.calls[1]["messages"][-1]["content"], "[Asha]: line 0")
        self.assertTrue(llm.calls[0]["messages"][0]["content"].endswith(conversation.AUTONOMY_NOTE))
        # Asha's second turn sees her own history: system, topic, her reply, Ravi's reply
        self.assertEqual([m["role"] for m in llm.calls[2]["messages"]], ["system", "user", "assistant", "user"])
        self.assertEqual(t["usage_totals"]["Asha"], {"input": 20, "output": 10})
        # Langfuse: one trace per turn + one conversation trace, gateway metadata on the session
        self.assertEqual(len(lf.turns), 4)
        self.assertEqual(len(lf.conversations), 1)
        self.assertEqual(lf.turns[0]["session"], "english__banking__00000-run1")
        self.assertEqual(llm.calls[1]["metadata"]["conversation_id"], "english__banking__00000-run1")
        self.assertEqual(llm.calls[1]["metadata"]["trace_name"], "turn-01-responder")

    def test_retry_then_success(self):
        llm = FakeLLM([ProviderError("HTTP 502"), reply("ok"), reply("ok2")])
        with mock.patch.object(conversation.time, "sleep"):
            t = conversation.run_conversation(spec(rounds=1), llm, None, "r")
        self.assertEqual(t["status"], "completed")
        self.assertEqual(len(t["transcript"]), 2)
        # the failed attempt left nothing behind in history
        self.assertEqual(len(llm.calls[1]["messages"]), 2)

    def test_gives_up_after_max_attempts(self):
        llm = FakeLLM([ProviderError("HTTP 504")] * conversation.MAX_ATTEMPTS)
        lf = FakeLangfuse()
        with mock.patch.object(conversation.time, "sleep"):
            t = conversation.run_conversation(spec(rounds=1), llm, lf, "r")
        self.assertEqual(t["status"], "error")
        self.assertIn("HTTP 504", t["error"])
        self.assertEqual(lf.conversations[0]["trace"]["status"], "error")

    def test_empty_reply_is_retried(self):
        llm = FakeLLM([reply(""), reply("real"), reply("x")])
        with mock.patch.object(conversation.time, "sleep"):
            t = conversation.run_conversation(spec(rounds=1), llm, None, "r")
        self.assertEqual(t["transcript"][0]["text"], "real")

    def test_web_search_tool_loop(self):
        call = [{"id": "c1", "type": "function", "function": {"name": "web_search",
                                                              "arguments": json.dumps({"query": "rbi rules"})}}]
        llm = FakeLLM([reply("question"), reply(None, call), reply("answer with source")])
        with mock.patch.dict("os.environ", {"TAVILY_API_KEY": "k"}), \
                mock.patch.object(web_search, "search", return_value=("RESULTS", {"query": "rbi rules"})):
            t = conversation.run_conversation(spec(rounds=1, search=True), llm, None, "r")
        self.assertEqual(t["transcript"][1]["text"], "answer with source")
        self.assertEqual(t["transcript"][1]["tool_calls"][0]["query"], "rbi rules")
        self.assertIsNone(llm.calls[0]["tools"])
        self.assertEqual(llm.calls[1]["tools"][0]["function"]["name"], "web_search")
        self.assertEqual(llm.calls[2]["messages"][-1], {"role": "tool", "tool_call_id": "c1", "content": "RESULTS"})

    def test_rejects_non_pair(self):
        s = spec()
        s["personas"].append(dict(s["personas"][0]))
        with self.assertRaises(ValueError):
            conversation.run_conversation(s, FakeLLM([]), None, "r")

    def test_sanitize(self):
        self.assertEqual(conversation.sanitize_text("a b​c"), "a bc")


class IOTests(unittest.TestCase):
    def test_trace_layout_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            t = trace.new_trace(spec())
            t["status"] = "completed"
            p = trace.write_trace(out, t)
            self.assertEqual(p, out / "english" / "banking" / "english__banking__00000.json")
            self.assertTrue(trace.already_completed(out, "english__banking__00000"))
            self.assertFalse(trace.already_completed(out, "missing"))

    def test_sharding(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.jsonl"
            f.write_text("".join(json.dumps({**spec(), "id": f"s{i}"}) + "\n" for i in range(10)))
            ids = lambda w, n, **kw: [s["id"] for s in select(iter_specs([f]), worker_id=w, total_workers=n,
                                                                offset=kw.get("offset", 0), limit=kw.get("limit"))]
            self.assertEqual(ids(1, 3), ["s1", "s4", "s7"])
            self.assertEqual(ids(0, 1, offset=2, limit=3), ["s2", "s3", "s4"])


if __name__ == "__main__":
    unittest.main()
