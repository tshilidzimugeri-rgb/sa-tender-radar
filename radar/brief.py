"""AI tender briefs: read the tender document and pull out what a bidder needs to know.

Uses a free LLM API. Set one of these environment variables (or Streamlit secrets):

    GEMINI_API_KEY   Google AI Studio, free tier   https://aistudio.google.com/apikey
    GROQ_API_KEY     Groq, free tier               https://console.groq.com/keys

LLM_MODEL overrides the default model for whichever provider is used. Briefs are cached in
data/briefs.json, so each tender is only sent to the API once.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
from datetime import datetime

import requests
from pypdf import PdfReader

from radar.etenders import HEADERS

log = logging.getLogger(__name__)

PROVIDERS = {
    "gemini": {"env": "GEMINI_API_KEY", "model": "gemini-2.5-flash", "max_chars": 60000},
    "groq": {"env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile", "max_chars": 18000},
}
MAX_PAGES = 20

PROMPT = """You are helping a small South African supplier decide whether to bid on a \
public tender. Read the tender details and document text below and reply with JSON only, \
using exactly these keys:

"summary": 2-3 plain-English sentences: what the buyer wants, where, and for how long.
"scope": up to 5 short bullet strings describing the deliverables.
"requirements": up to 8 short strings listing what a bidder must have or submit to \
qualify (e.g. CIDB grading, CSD registration, tax compliance, professional registration, \
experience, site visit).
"cidb_grading": the CIDB grading required, e.g. "6CE or higher", or null.
"preference": the preference point system and specific goals, e.g. "80/20, 20 points \
for black ownership", or null.
"evaluation": how bids are scored, including any functionality threshold, or null.
"contract_period": the contract duration, or null.
"estimated_value": the budget or estimated value if stated, or null.
"watch_out": up to 4 short strings with things that could disqualify a bidder or are \
easy to miss (compulsory briefings, short deadlines, unusual conditions).

Only use facts from the text. Use null or [] when something is not stated. Do not \
invent values.

TENDER DETAILS
{details}

DOCUMENT TEXT
{document}
"""

LIST_KEYS = ("scope", "requirements", "watch_out")
TEXT_KEYS = ("summary", "cidb_grading", "preference", "evaluation", "contract_period",
             "estimated_value")


def _secret(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def available_provider() -> str | None:
    for name, cfg in PROVIDERS.items():
        if _secret(cfg["env"]):
            return name
    return None


def document_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        return ""
    reader = PdfReader(io.BytesIO(resp.content))
    pages = []
    for page in reader.pages[:MAX_PAGES]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # some tender PDFs have broken fonts
            continue
    text = re.sub(r"[ \t]+", " ", "\n".join(pages))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _call_gemini(prompt: str, key: str, model: str) -> str:
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.1,
                                   "responseMimeType": "application/json"}},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(prompt: str, key: str, model: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0.1,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": prompt}]},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _clean(raw: dict) -> dict:
    out = {}
    for key in LIST_KEYS:
        value = raw.get(key) or []
        if isinstance(value, str):
            value = [value]
        out[key] = [str(v).strip() for v in value if str(v).strip()][:8]
    for key in TEXT_KEYS:
        value = raw.get(key)
        out[key] = None if value in (None, "", "null", "N/A") else str(value).strip()
    return out


def generate(tender: dict, provider: str | None = None) -> dict:
    """Build a brief for one tender (a row of the tenders table as a dict)."""
    provider = provider or available_provider()
    if not provider:
        raise RuntimeError("No LLM API key set (GEMINI_API_KEY or GROQ_API_KEY)")
    cfg = PROVIDERS[provider]
    model = _secret("LLM_MODEL") or cfg["model"]

    docs = json.loads(tender.get("documents") or "[]")
    text, source = "", "listing"
    for doc in docs[:3]:
        try:
            text = document_text(doc["url"])
        except Exception as exc:
            log.warning("could not read %s: %s", doc.get("name"), exc)
            continue
        if len(text) > 500:
            source = doc["name"]
            break

    details = "\n".join(f"{k}: {tender.get(k)}" for k in (
        "reference", "description", "buyer", "province", "delivery", "type", "closes",
        "briefing_date", "briefing_venue", "conditions") if tender.get(k))
    prompt = PROMPT.format(details=details,
                           document=text[:cfg["max_chars"]] or "(no readable document)")
    call = _call_gemini if provider == "gemini" else _call_groq
    raw = call(prompt, _secret(cfg["env"]), model)
    raw = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()
    brief = _clean(json.loads(raw))
    brief.update(source=source, model=f"{provider}/{model}",
                 generated=datetime.now().strftime("%Y-%m-%d %H:%M"))
    return brief
