"""Rank open tenders against a business profile.

A profile is what a supplier would tell a bid consultant: the sectors they work in, the
provinces they can deliver to, and a few phrases that describe their work. The score is a
weighted blend of three signals, each between 0 and 1:

- sector:   1 if the classifier put the tender in one of the profile's sectors, 0.5 if it
            was the model's second choice, otherwise 0
- keywords: TF-IDF cosine similarity between the profile phrases and the tender text,
            rescaled so a strong phrase match reaches 1
- province: 1 inside the profile's provinces, 0.6 for national tenders, otherwise 0
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

WEIGHTS = {"sector": 0.40, "keywords": 0.30, "province": 0.30}
NATIONAL = "National"


@dataclass
class Profile:
    name: str = "My business"
    sectors: list[str] = field(default_factory=list)
    provinces: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    email: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _text(df: pd.DataFrame) -> pd.Series:
    return (df["description"].fillna("") + " " + df["category"].fillna("") + " "
            + df["buyer"].fillna("")).str.lower()


def score(df: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Return df with score (0-100), score parts and a short reason, best first."""
    out = df.copy()
    text = _text(out)

    if profile.sectors:
        sector = np.where(out["sector"].isin(profile.sectors), 1.0,
                          np.where(out["sector_2"].isin(profile.sectors), 0.5, 0.0))
    else:
        sector = np.full(len(out), np.nan)

    phrases = [p.strip().lower() for p in profile.keywords if p.strip()]
    if phrases and len(out):
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        docs = vec.fit_transform(text)
        sims = linear_kernel(vec.transform(phrases), docs)  # phrases x tenders
        best = sims.max(axis=0)
        exact = np.array([any(p in t for p in phrases) for t in text])
        keywords = np.clip(np.maximum(best / 0.35, exact * 0.9), 0, 1)
    else:
        keywords = np.full(len(out), np.nan)

    if profile.provinces:
        province = np.where(out["province"].isin(profile.provinces), 1.0,
                            np.where(out["province"].eq(NATIONAL), 0.6, 0.0))
    else:
        province = np.full(len(out), np.nan)

    parts = pd.DataFrame({"sector": sector, "keywords": keywords, "province": province},
                         index=out.index)
    weights = pd.Series(WEIGHTS)
    used = parts.notna()
    total = (parts.fillna(0) * weights).sum(axis=1)
    norm = (used * weights).sum(axis=1).replace(0, np.nan)
    out["score"] = (100 * total / norm).fillna(0).round(0)
    for col in parts:
        out[f"fit_{col}"] = parts[col]

    if profile.exclude:
        rx = re.compile("|".join(re.escape(e.strip().lower()) for e in profile.exclude
                                 if e.strip()))
        if rx.pattern:
            out.loc[text.str.contains(rx), "score"] = 0

    out["reason"] = [_reason(r, profile) for r in out.itertuples()]
    return out.sort_values(["score", "closes"], ascending=[False, True])


def _reason(row, profile: Profile) -> str:
    bits = []
    if profile.sectors and row.fit_sector == 1:
        bits.append(row.sector)
    elif profile.sectors and row.fit_sector == 0.5:
        bits.append(f"possibly {row.sector_2}")
    if profile.keywords and row.fit_keywords >= 0.6:
        bits.append("matches your keywords")
    if profile.provinces and row.fit_province == 1:
        bits.append(row.province)
    elif profile.provinces and row.fit_province > 0:
        bits.append("national")
    return " · ".join(bits)
