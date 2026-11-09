# datagen-conversation

Generates multilingual, **2-person** synthetic advisory conversations: an
*asker* persona (a farmer, a bank customer, a student, …) with a real
problem talks with a *responder* persona (an advisor or expert who can use
live web search) for 10 rounds (20 turns). Each conversation is saved as a
local trace JSON and exported to **Langfuse**.

There are two steps:

1. **Task bank (the prompts).** `convgen.task_bank` builds conversation specs
   from 20 domains × 42 languages: the opening question plus both personas'
   system prompts. No model calls are made. Output is deterministic, so the
   same command always produces the same files.
2. **Conversations (the synthetic data).** `convgen.run` plays each spec out
   against the LLM gateway, writes `data/traces/<language>/<domain>/<id>.json`
   and sends the traces to Langfuse as the conversation runs.

---

## Repo structure

```
datagen-conversation/
├── convgen/
│   ├── domains.py        # 20 domains: roles, system-prompt templates, issues, modifiers, which role may search
│   ├── languages.py      # 42 languages: reply instruction, places, staples, currency, regulator, person names
│   ├── task_bank.py      # STEP 1: domains × languages → data/task_bank/<domain>/<language>.jsonl
│   ├── run.py            # STEP 2: CLI — run specs, write traces, export to Langfuse
│   ├── conversation.py   #   the 2-person turn loop, retries, web_search tool loop
│   ├── llm.py            #   OpenAI-compatible chat-completions client (Soket gateway)
│   ├── web_search.py     #   web_search tool (Tavily)
│   ├── langfuse.py       #   Langfuse ingestion-API client (sessions, turn + conversation traces)
│   ├── trace.py          #   local trace JSON schema, file layout, resume check
│   ├── upload_traces.py  # push existing local traces to Langfuse (backfill / re-send)
│   ├── status.py         # tally traces by status / language / domain
│   └── env.py            # loads .env
├── data/
│   ├── task_bank/        # generated specs (git-ignored, regenerate with step 1)
│   └── traces/           # generated conversations (git-ignored)
├── examples/
│   ├── task_bank/agriculture/bengali.jsonl                          # 3 sample specs
│   └── traces/bengali/agriculture/bengali__agriculture__00004.json  # 1 sample conversation (20 turns)
├── k8s/conv_gen_job.yaml # 20-pod Indexed Job that shards data/task_bank across pods
├── tests/test_convgen.py # offline tests (no network, no keys)
├── .env.example
├── requirements.txt
└── pyproject.toml
```

---

## Setup

```bash
git clone https://github.com/saravanan-soket/datagen-conversation.git
cd datagen-conversation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
```

`.env`:

| Variable | Needed for | Notes |
|---|---|---|
| `SOKET_BASE_URL` | step 2 | Gateway URL ending in `/v1`. In the cluster this is the `llm-logging/litellm-proxy` NodePort `31184` on any **Ready** node. |
| `SOKET_API_KEY` | step 2 | Falls back to `TENSORSTUDIO_API_KEY`, `LOOP_API_KEY` or `OPENAI_API_KEY`. |
| `TAVILY_API_KEY` | optional | Enables `web_search` for responder personas. If unset, they run without the tool. |
| `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | step 2 | `llm-logging/langfuse` NodePort `30395`. Not needed with `--no-langfuse`. |
| `CONVGEN_MAX_TOKENS`, `CONVGEN_TEMPERATURE` | optional | Sent with every request when set. |

Check that the endpoints are up:

```bash
curl -s -o /dev/null -w '%{http_code}\n' "$SOKET_BASE_URL/models" -H "Authorization: Bearer $SOKET_API_KEY"
curl -s "$LANGFUSE_BASE_URL/api/public/health"
kubectl get svc -n llm-logging litellm-proxy langfuse    # in-cluster: NodePorts
kubectl get nodes -o wide                                # pick a Ready node's INTERNAL-IP
```

---

## Step 1: generate the prompts (task bank)

```bash
# list the language and domain slugs
python -m convgen.task_bank --list

# the shards behind the original multilingual_task_bank (10,000 specs each)
python -m convgen.task_bank --domains agriculture \
  --languages bengali,hindi,kannada,marathi,odia,tamil,telugu
python -m convgen.task_bank --domains banking --languages english

# a small bank for trying things out
python -m convgen.task_bank --domains agriculture,banking --languages hindi,english --count 20

# everything: 20 domains × 42 languages × 10,000 = 8.4M specs
python -m convgen.task_bank

# overwrite shards that already exist
python -m convgen.task_bank --domains agriculture --languages hindi --force
```

The output is `data/task_bank/<domain>/<language>.jsonl`, one spec per line
(see `examples/task_bank/agriculture/bengali.jsonl`):

```json
{
  "id": "bengali__agriculture__00000",
  "topic": "My rice crop near Dhaka is showing a pest infestation. What should I do?\n\nWrite your entire reply in Bengali ...",
  "rounds_per_persona": 10,
  "personas": [
    {"name": "Arup",    "model": "soket/qwen3-8-27b", "system_prompt": "You are a farmer near Dhaka ...", "allow_web_search": false},
    {"name": "Mousumi", "model": "soket/gpt",         "system_prompt": "You are an agricultural extension advisor ...", "allow_web_search": true}
  ]
}
```

**How a spec is built.** A random generator seeded with
`"<language>:<domain>:<index>"` picks the issue, the modifier, the topic
template and (for region-sensitive domains) a local place and staple crop or
product. It also picks two names from the language's alternating male/female
name list, and every system prompt states both personas' genders. Languages
with grammatical gender therefore use the correct forms. Every prompt ends
with the language's "reply only in X" instruction. Personas are always asked
by `soket/qwen3-8-27b` and answered by `soket/gpt`; to change that, edit
`ASKER_MODEL` and `RESPONDER_MODEL` in `task_bank.py`.

To add a domain or language, add an entry to `domains.py` or `languages.py`
and run step 1 again.

---

## Step 2: generate the synthetic conversations

```bash
# smoke test: one conversation, 2 rounds, local JSON only
python -m convgen.run --spec data/task_bank/agriculture/hindi.jsonl --limit 1 --rounds 2 --no-langfuse

# one shard, 4 conversations in parallel, exported to Langfuse
python -m convgen.run --spec data/task_bank/agriculture/hindi.jsonl --workers 4

# the whole bank (directories are searched recursively)
python -m convgen.run --spec data/task_bank --workers 8 --run-id agri-v1 --tag agriculture

# a slice: specs 100..149 of a shard
python -m convgen.run --spec data/task_bank/banking/english.jsonl --offset 100 --limit 50

# shard across 4 machines (run one command on each, worker-id 0..3)
python -m convgen.run --spec data/task_bank --worker-id 0 --total-workers 4 --workers 4

# progress
python -m convgen.status --traces data/traces
```

| Flag | Meaning |
|---|---|
| `--spec PATH` | A `.json` spec, a `.jsonl` shard, or a directory of them. |
| `--out DIR` | Directory for local traces (default `data/traces`). |
| `--workers N` | Number of conversations run in parallel. |
| `--limit N` / `--offset N` | Run N specs / skip the first N. |
| `--worker-id I --total-workers N` | Take every Nth spec starting at I. Defaults come from `$JOB_COMPLETION_INDEX` / `$TOTAL_WORKERS`. |
| `--run-id ID` | Langfuse run id; part of every session id. Defaults to a UTC timestamp. |
| `--tag T` | Extra Langfuse tag (repeatable). |
| `--rounds N` | Override `rounds_per_persona`, e.g. for quick tests. |
| `--no-langfuse` | Write local JSON only. |
| `--rerun` | Re-run conversations that already completed. |

**Resuming.** A conversation whose local trace already has
`status: completed` is skipped. After an interruption, run the same command
again and it continues where it stopped. Conversations that errored are
retried on the next run.

### How a conversation runs

- The asker receives the spec's `topic` as its first message. After that,
  each persona receives the other's last line as `[Name]: text`.
- Each persona keeps its own history: its system prompt plus a fixed suffix
  ("conversational length, stay in character, never say you are an AI, don't
  re-greet"), then the conversation so far from its own point of view.
- A persona with `allow_web_search` gets a `web_search` tool (Tavily, top 5
  results plus a summary). It may search several times before replying.
- A failed turn (HTTP error, timeout, gateway error, empty reply) is retried
  up to 5 times, waiting 1s, 2s, 4s and 8s between attempts. If all 5 fail,
  the conversation is saved with `status: error`.
- Invisible non-standard spaces (U+202F, U+00A0, U+2009, U+2060, U+200B),
  which some models emit in non-Latin scripts, are converted to plain spaces
  or removed.

### Local trace format

Saved at `data/traces/<language>/<domain>/<id>.json` (see `examples/traces/`):

```json
{
  "conversation_id": "bengali__agriculture__00004",
  "topic": "...",
  "created_at": "2026-...",
  "personas": [{"name": "...", "model": "soket/qwen3-8-27b", "allow_web_search": false}, {"...": "..."}],
  "transcript": [
    {"round": 1, "persona": "...", "text": "...", "tool_calls": [], "usage": {"input": 512, "output": 88}},
    {"round": 1, "persona": "...", "text": "...",
     "tool_calls": [{"tool": "web_search", "query": "...", "result_preview": "..."}],
     "usage": {"input": 1430, "output": 160}}
  ],
  "status": "completed",
  "error": null,
  "usage_totals": {"<asker>": {"input": 0, "output": 0}, "<responder>": {"input": 0, "output": 0}}
}
```

---

## Traces in Langfuse

Each conversation gets **its own Langfuse session**, with the id
`<conversation_id>-<run_id>` (e.g. `hindi__agriculture__00042-agri-v1`).

| Trace | Input → Output | Contents |
|---|---|---|
| `conversation` (id = session id) | the whole chat as `[{role, content}]` (asker = `user`, responder = `assistant`) → `{status, error, turns, usage_totals}` | This is the dataset row. Metadata holds the topic, personas and number of web searches. |
| `turn-NN-asker` / `turn-NN-responder` | the incoming line → the reply | One `llm-response` generation per model request (exact messages, model, token usage, latency), a `thinking` span when the model returned reasoning, and a `tool:web_search` span per search. |

Tags: `conversation-gen`, `run:<run_id>`, `asker`/`responder`/`conversation`,
`completed`/`error`, plus any `--tag` values.

The gateway (LiteLLM) also writes its own Langfuse trace for every model
call. Each request carries `metadata.conversation_id`, `trace_name` and
`tags`, so those gateway traces land in the **same session** instead of a
random one.

Sending to Langfuse never blocks generation. A failed batch is retried 4
times, then logged, and the run continues. At the end, the run prints how
many batches failed and the command to re-send them.

### Uploading existing traces

This covers traces made with `--no-langfuse`, batches that failed to send,
and traces from the original Rust engine (same JSON schema):

```bash
python -m convgen.upload_traces --traces data/traces
python -m convgen.upload_traces --traces /path/to/multilingual_dataset/traces --run-id backfill-v1 --tag rust-engine
python -m convgen.upload_traces --traces data/traces --status all --limit 100
```

Uploads use fixed trace ids derived from the conversation id and run id, so
uploading the same files again updates the existing traces instead of
creating duplicates. A local trace stores only each turn's final text and
token counts, so uploaded turn traces don't include the raw model requests.

---

## Kubernetes (20 pods)

```bash
# on the host: build the bank and fill .env first (pods mount the repo from the host)
python -m convgen.task_bank --domains agriculture --languages hindi,tamil
kubectl apply -f k8s/conv_gen_job.yaml
kubectl get pods -l app=conv-gen
kubectl logs -f job/conv-gen-20pods          # or a specific pod
python -m convgen.status --traces data/traces
kubectl delete job conv-gen-20pods           # when done
```

Edit `REPO` and the `hostPath` in the YAML to match where the repo is
checked out. Pod *i* runs the specs at global indices *i*, *i+20*, *i+40*, ….

---

## Tests

```bash
python -m unittest discover -s tests -v
```

These run offline with a fake LLM and a fake Langfuse. They cover turn order
and message framing, per-persona history, retries and giving up, empty-reply
retries, the web-search tool loop, rejection of specs that don't have exactly
2 personas, space normalisation, trace layout and resume, sharding, and
deterministic pair-only spec generation.

---

## Where this came from

This repo is the 2-person conversation part of the `synthetic_datagen`
monorepo, extracted on its own:

| Original | Here |
|---|---|
| `headless-harness/conversation_gen/multilingual_gen.py` (made `artifacts/multilingual_task_bank/`) | `convgen/task_bank.py`. Output is identical: the bengali, hindi and english shards were checked to match byte for byte. |
| `headless-harness/conversation_gen/domains.py`, `languages.py` | `convgen/domains.py` (only the 20 two-person domains; the 8 four-person "group" domains were removed) and `convgen/languages.py` |
| `crates/loop-conversation-gen` (Rust; made `artifacts/multilingual_dataset/traces/`) | `convgen/conversation.py`, `llm.py`, `web_search.py`, `trace.py`, `run.py`. Same prompts, message framing, retry policy, text cleanup and trace schema, rewritten against the gateway's OpenAI-compatible API directly instead of the `loop-agent` harness. |
| `headless-harness/conversation_gen/k8s_worker.py` + `k8s/multilingual_conv_gen_job.yaml` | `--worker-id/--total-workers` in `run.py` + `k8s/conv_gen_job.yaml` |
| *(none: the Rust engine only wrote local JSON, and Langfuse got only the gateway's per-call traces)* | `convgen/langfuse.py` + `upload_traces.py` |

Differences from the Rust engine:
- An empty model reply is now retried instead of being saved as an empty turn.
- A failed attempt no longer stays in the persona's history; the retry resends the original line.
- Requests carry gateway metadata so per-call traces group by conversation.
- Langfuse export is built in.

Note: in the original task bank, `cybersecurity/english.jsonl` is a
4-person group conversation and isn't supported here. In this repo,
`cybersecurity` is not a domain.
