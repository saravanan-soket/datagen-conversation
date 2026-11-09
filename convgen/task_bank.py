"""task_bank.py — generate the language x domain task bank (the prompt files).

One JSONL shard per (domain, language) pair: `<out>/<domain>/<language>.jsonl`,
one ConversationSpec per line. Each shard is consumed directly by
`python -m convgen.run --spec <file>.jsonl`.

Uniqueness without a model call per spec: each spec is built by deterministic
pseudo-random sampling (seeded on language+domain+index) across independent
axes — topic template, issue, modifier, place/staple/currency (region-
sensitive domains only), and persona names. Same seed, same output: rerunning
this reproduces an existing shard byte for byte.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from convgen.domains import DOMAINS, Domain
from convgen.languages import LANGUAGES, Language

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "task_bank"

ASKER_MODEL = "soket/qwen3-8-27b"
RESPONDER_MODEL = "soket/gpt"
PAIR_MODELS = [ASKER_MODEL, RESPONDER_MODEL]

PAIR_ROUNDS = 10  # x2 personas = 20 total turns


def build_spec(lang: Language, domain: Domain, index: int) -> dict:
    rng = random.Random(f"{lang.slug}:{domain.slug}:{index}")

    fmt = {
        "issue": rng.choice(domain.issues),
        "modifier": rng.choice(domain.modifiers),
    }
    if domain.region_sensitive:
        fmt.update(
            place=rng.choice(lang.places),
            staple=rng.choice(lang.staples),
            currency=lang.currency,
            regulator=lang.regulator,
            region=lang.region,
        )

    template = rng.choice(domain.topic_templates)
    topic = f"{template.format(**fmt)}\n\n{lang.instruction}"

    # `person_names` is authored as a strict [male, female, male, female, ...]
    # sequence per language (see languages.py) — sample indices, not names, so
    # each pick keeps its gender via index parity. Without this, the model has
    # no gender signal beyond the name itself and was observed defaulting to
    # male-gendered terms of address (e.g. Tamil "அண்ணா") for female personas
    # in languages with grammatical/honorific gender.
    name_idxs = rng.sample(range(len(lang.person_names)), k=len(domain.roles))
    picks = [(lang.person_names[i], "male" if i % 2 == 0 else "female") for i in name_idxs]
    roster_note = " ".join(f"{n} is {g}." for n, g in picks)

    personas = []
    for i in range(len(domain.roles)):
        desc = domain.role_descs[i].format(**fmt)
        name, gender = picks[i]
        gender_note = (
            f"Your name is {name} and you are {gender}. {roster_note} If {lang.name} "
            f"grammatically marks gender (verb forms, adjectives, or terms of address such as "
            f"brother/sister-style honorifics), use forms consistent with each person's actual "
            f"gender as given above — never default to male forms for a female persona or vice versa."
        )
        personas.append({
            "name": name,
            "model": PAIR_MODELS[i],
            "system_prompt": f"{desc} {gender_note} {lang.instruction}",
            "allow_web_search": domain.search_roles[i],
        })

    return {
        "id": f"{lang.slug}__{domain.slug}__{index:05d}",
        "topic": topic,
        "rounds_per_persona": PAIR_ROUNDS,
        "personas": personas,
    }


def write_shard(lang: Language, domain: Domain, count: int, out_dir: Path, *, force: bool) -> Path:
    shard_dir = out_dir / domain.slug
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"{lang.slug}.jsonl"
    if path.is_file() and not force:
        return path
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for i in range(count):
            spec = build_spec(lang, domain, i)
            fh.write(json.dumps(spec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the multilingual x multi-domain task bank")
    parser.add_argument("--count", type=int, default=10000, help="Conversations per (language, domain) pair")
    parser.add_argument("--languages", default="all", help="Comma-separated language slugs, or 'all'")
    parser.add_argument("--domains", default="all", help="Comma-separated domain slugs, or 'all'")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="Regenerate shards that already exist")
    parser.add_argument("--list", action="store_true", help="List language and domain slugs and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("languages:", ",".join(l.slug for l in LANGUAGES))
        print("domains:  ", ",".join(d.slug for d in DOMAINS))
        return 0

    langs = LANGUAGES if args.languages == "all" else [
        l for l in LANGUAGES if l.slug in set(args.languages.split(","))
    ]
    doms = DOMAINS if args.domains == "all" else [
        d for d in DOMAINS if d.slug in set(args.domains.split(","))
    ]
    if not langs or not doms:
        parser.error("no matching languages/domains (see --list)")

    total_combos = len(langs) * len(doms)
    total_specs = total_combos * args.count
    print(
        f"Generating {total_specs:,} specs across {total_combos:,} shards "
        f"({len(langs)} languages x {len(doms)} domains x {args.count:,} each) -> {args.out}"
    )

    start = time.monotonic()
    done = 0
    for domain in doms:
        for lang in langs:
            write_shard(lang, domain, args.count, args.out, force=args.force)
            done += 1
            if done % 25 == 0 or done == total_combos:
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_combos - done) / rate if rate > 0 else 0
                print(f"  {done:,}/{total_combos:,} shards ({done * args.count:,} specs) "
                      f"— {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    print(f"Done: {total_combos:,} shards, {total_specs:,} specs, {time.monotonic() - start:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
