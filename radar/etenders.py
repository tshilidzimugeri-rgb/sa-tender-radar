"""Client for the public eTenders portal (www.etenders.gov.za).

The portal's tender list is backed by a JSON endpoint that returns every tender with a
given status in a single response, with province, contact and briefing details that the
National Treasury OCDS API leaves out. No key is needed.

    status=1  currently advertised (about 2,000 tenders, ~5 MB)
    status=4  cancelled (about 25,000, ~50 MB) - used as classifier training text
"""
from __future__ import annotations

import json
import logging
import time
from urllib.parse import quote

import pandas as pd
import requests

URL = "https://www.etenders.gov.za/Home/TenderOpportunities/"
DOWNLOAD = "https://www.etenders.gov.za/home/Download?blobName={blob}&downloadedFileName={name}"
OPEN, CANCELLED = 1, 4
HEADERS = {"User-Agent": "sa-tender-radar (+https://github.com/tshilidzimugeri-rgb/sa-tender-radar)"}
log = logging.getLogger(__name__)

COLUMNS = [
    "id", "reference", "type", "description", "category", "buyer", "province",
    "delivery", "published", "closes", "briefing", "briefing_compulsory",
    "briefing_date", "briefing_venue", "contact", "email", "phone", "conditions",
    "esubmission", "documents",
]
DATE_COLUMNS = ["published", "closes", "briefing_date"]


def fetch_raw(status: int = OPEN, timeout: int = 600) -> list[dict]:
    for attempt in range(3):
        try:
            resp = requests.get(URL, params={"status": status}, headers=HEADERS,
                                timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("eTenders request failed (%s), attempt %d", exc, attempt + 1)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("eTenders is not responding")


def _clean(value) -> str:
    text = " ".join(str(value or "").split())
    return "" if text.lower() in {"<not available>", "n/a", "none", "null", "-"} else text


def _yes(value) -> bool:
    return str(value).strip().lower() in {"yes", "true"}


def _documents(sd) -> str:
    docs = []
    for d in sd or []:
        if d.get("active") is False:
            continue
        name = d.get("fileName") or "document"
        blob = f"{d['supportDocumentID']}{d.get('extension') or ''}"
        docs.append({"name": name, "url": DOWNLOAD.format(blob=blob, name=quote(name))})
    return json.dumps(docs, ensure_ascii=False)


def flatten(t: dict) -> dict:
    return {
        "id": t["id"],
        "reference": _clean(t.get("tender_No")),
        "type": _clean(t.get("type")),
        "description": _clean(t.get("description")),
        "category": _clean(t.get("category")),
        "buyer": _clean(t.get("department")),
        "province": _clean(t.get("province")),
        "delivery": _clean(t.get("delivery")),
        "published": t.get("date_Published"),
        "closes": t.get("closing_Date"),
        "briefing": _yes(t.get("bf")),
        "briefing_compulsory": _yes(t.get("bc")),
        "briefing_date": t.get("compulsory_briefing_session"),
        "briefing_venue": _clean(t.get("briefingVenue")),
        "contact": _clean(t.get("contactPerson")),
        "email": _clean(t.get("email")),
        "phone": _clean(t.get("telephone")),
        "conditions": _clean(t.get("conditions")),
        "esubmission": bool(t.get("eSubmission")),
        "documents": _documents(t.get("sd")),
    }


def to_frame(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([flatten(t) for t in raw], columns=COLUMNS)
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df.loc[df[col].dt.year < 2000, col] = pd.NaT  # the portal uses 0001-01-01 for "none"
    return df.drop_duplicates("id", keep="last")


def fetch_open() -> pd.DataFrame:
    return to_frame(fetch_raw(OPEN))
