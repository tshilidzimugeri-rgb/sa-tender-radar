"""Paths and loaders for the data files committed to the repo."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from radar.etenders import COLUMNS, DATE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TENDERS = DATA / "tenders.csv.gz"      # every tender seen in a daily snapshot
CORPUS = DATA / "corpus.csv.gz"        # classifier training text
BRIEFS = DATA / "briefs.json"          # AI briefs, keyed by tender id
STATUS = DATA / "status.json"          # last run summary and model report
PROFILES = ROOT / "profiles.json"

TENDER_DATES = DATE_COLUMNS + ["first_seen"]


def load_tenders() -> pd.DataFrame:
    if not TENDERS.exists():
        return pd.DataFrame(columns=COLUMNS + ["first_seen"])
    df = pd.read_csv(TENDERS, keep_default_na=False, low_memory=False)
    for col in TENDER_DATES:
        df[col] = pd.to_datetime(df[col].replace("", None), errors="coerce")
    for col in ("briefing", "briefing_compulsory", "esubmission"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    if "sector_confidence" in df:
        df["sector_confidence"] = pd.to_numeric(df["sector_confidence"], errors="coerce")
    return df


def save_tenders(df: pd.DataFrame) -> None:
    DATA.mkdir(exist_ok=True)
    df.to_csv(TENDERS, index=False, compression={"method": "gzip", "mtime": 0})


def load_corpus() -> pd.DataFrame:
    return pd.read_csv(CORPUS, keep_default_na=False)


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
