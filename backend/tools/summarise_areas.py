"""Write one short, neutral summary per area with an AI model -- offline.

Providers (set MEILI_AI_PROVIDER):
  github     (default) GitHub Models -- free, rate-limited; uses GITHUB_TOKEN,
             which GitHub Actions provides automatically.
  ollama     A model running on your own computer (https://ollama.com) -- free,
             offline. Default model llama3.1:8b.
  anthropic  Claude via the Anthropic API -- paid per use, needs ANTHROPIC_API_KEY.
Override the model with MEILI_SUMMARY_MODEL.

For every city in data/social_context/, records are grouped per area exactly
as the backend groups them. Each group is sent to Claude with the records'
text ONLY, and Claude must return one sentence (max 30 words) that adds no
facts of its own. Results go to data/social_context/<city>/area_summaries.json
and are reused until the records of that area change.

Run:  python tools/summarise_areas.py            (inside GitHub Actions)
      MEILI_AI_PROVIDER=ollama python tools/summarise_areas.py   (on a laptop)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import social_context_analysis as sc  # noqa: E402

PROVIDER = os.getenv("MEILI_AI_PROVIDER", "gemini" if os.getenv("GEMINI_API_KEY") else "github").lower()
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "github": "openai/gpt-4o-mini",   # see github.com/marketplace/models
    "ollama": "llama3.1:8b",
    "anthropic": "claude-haiku-5-5",
}
MODEL = os.getenv("MEILI_SUMMARY_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))
OPENAI_STYLE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "github": "https://models.github.ai/inference/chat/completions",
    "ollama": os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1/chat/completions",
}
MAX_WORDS = 30

SYSTEM = (
    "You summarise local safety evidence for a walking-safety app. "
    "Use ONLY the records provided. Never add facts, places, numbers, causes or "
    "advice that are not in the records. Do not exaggerate or soften. If the "
    "records disagree, say so briefly. Write one neutral sentence of at most "
    f"{MAX_WORDS} words, in English, about walking in the area. "
    'Reply with JSON only: {"summary": "...", "verdict": "concerns|positive|mixed"}'
)


def area_groups(city_id: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in sc.load_opinions(city_id):
        if not record.get("rating_eligible", False):
            continue
        groups.setdefault(sc._area_key(record), []).append(record)
    return groups


def fingerprint(records: list[dict]) -> str:
    ids = sorted(str(r.get("record_id")) for r in records)
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def _credentials() -> str:
    if PROVIDER == "github":
        return os.getenv("GITHUB_TOKEN", "")
    if PROVIDER == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    if PROVIDER == "groq":
        return os.getenv("GROQ_API_KEY", "")
    if PROVIDER == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY", "")
    return "ollama"  # local, no key


def _complete(user_text: str, key: str) -> str:
    if PROVIDER == "anthropic":
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": MODEL, "max_tokens": 1024, "system": SYSTEM,
                  "messages": [{"role": "user", "content": user_text}]},
            timeout=60,
        )
        response.raise_for_status()
        return "".join(b.get("text", "") for b in response.json().get("content", []) if b.get("type") == "text")

    response = None
    for wait in (0, 10, 20, 40):          # free tiers answer 429 when asked too often
        time.sleep(wait)
        response = requests.post(
            OPENAI_STYLE_URLS[PROVIDER],
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": MODEL, "max_tokens": 1024, "temperature": 0.2,
                  **({"reasoning_effort": "low"} if PROVIDER == "gemini" else {}),
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": user_text}]},
            timeout=120,
        )
        if response.status_code not in (429, 500, 502, 503, 504):
            break
        print(f"    (model busy: HTTP {response.status_code}, waiting before retrying)")
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask_model(area: str, records: list[dict], key: str) -> dict:
    lines = [
        f"- [{r.get('overall_sentiment')}, {r.get('evidence_tier')}, {r.get('post_date') or 'undated'}] "
        f"{r.get('opinion_summary')}"
        for r in records
    ]
    text = _complete(f"Area: {area}\nRecords:\n" + "\n".join(lines), key)
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(cleaned[cleaned.find("{"):cleaned.rfind("}") + 1])
    summary = " ".join(str(data["summary"]).split())
    if len(summary.split()) > MAX_WORDS + 5:
        raise ValueError(f"summary too long: {summary}")
    if data.get("verdict") not in ("concerns", "positive", "mixed"):
        raise ValueError(f"bad verdict: {data}")
    return {"summary": summary, "verdict": data["verdict"]}


def main() -> int:
    if PROVIDER not in DEFAULT_MODELS:
        print(f"Unknown MEILI_AI_PROVIDER '{PROVIDER}' (use github, ollama or anthropic).")
        return 1
    api_key = _credentials()
    if not api_key:
        print(f"No credentials for provider '{PROVIDER}' (GITHUB_TOKEN or ANTHROPIC_API_KEY).")
        return 1
    print(f"Provider: {PROVIDER}, model: {MODEL}")
    failures = 0
    for city_id in sc.registered_city_ids():
        path = sc.DATA_ROOT / city_id / "area_summaries.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        output = {}
        for key, records in sorted(area_groups(city_id).items()):
            fp = fingerprint(records)
            if existing.get(key, {}).get("fingerprint") == fp:
                output[key] = existing[key]          # records unchanged: reuse
                continue
            name = sc._display_name(records)
            try:
                result = ask_model(name, records, api_key)
            except Exception as error:  # noqa: BLE001
                failures += 1
                print(f"  ! {city_id}/{name}: {error}")
                continue
            output[key] = {
                **result,
                "area": name,
                "record_ids": sorted(str(r.get("record_id")) for r in records),
                "fingerprint": fp,
                "model": MODEL,
                "generated_at": time.strftime("%Y-%m-%d"),
                "needs_review": True,   # a person should read new summaries once
            }
            print(f"  + {city_id}/{name}: {result['summary']}")
            time.sleep(6)  # stay well inside free-tier per-minute limits
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
        print(f"{city_id}: {len(output)} areas written")
    if failures:
        print(f"::warning::{failures} areas could not be summarised this time (busy AI service). "
              "Their summaries will be written on the next run; finished areas are kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
