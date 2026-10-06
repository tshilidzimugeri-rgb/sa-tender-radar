"""The daily run: fetch, classify, match, brief, notify. Called by scripts/run_daily.py."""
from __future__ import annotations

import logging
import re
from datetime import timedelta

import pandas as pd

from radar import brief as briefs_mod
from radar.classify import apply, train
from radar.etenders import fetch_open
from radar.match import Profile, score
from radar.notify import deliver
from radar.store import (BRIEFS, PROFILES, STATUS, load_corpus, load_json, load_tenders,
                         save_json, save_tenders)

log = logging.getLogger(__name__)

KEEP_CLOSED_DAYS = 180
MIN_SCORE = 55
MAX_PER_DIGEST = 10
NOTICE_RX = re.compile(r"^\s*(?:publication of (?:the )?(?:names|bidders)|notice of award|"
                       r"award(?:ed)? (?:notice|of contract)|names of bidders)", re.I)


def now_sa() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Africa/Johannesburg").tz_localize(None)


def merge_snapshot(archive: pd.DataFrame, snap: pd.DataFrame, today) -> pd.DataFrame:
    seen = dict(zip(archive["id"], archive["first_seen"])) if len(archive) else {}
    snap = snap.copy()
    # On the very first run there is no history, so fall back to the publish date.
    unseen = snap["published"].fillna(today) if archive.empty else pd.Timestamp(today)
    snap["first_seen"] = snap["id"].map(seen).fillna(unseen)
    rest = archive[~archive["id"].isin(snap["id"])]
    merged = pd.concat([rest, snap], ignore_index=True) if len(rest) else snap
    cutoff = pd.Timestamp(today) - timedelta(days=KEEP_CLOSED_DAYS)
    return merged[merged["closes"].isna() | (merged["closes"] >= cutoff)]


def open_tenders(df: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    return df[(df["closes"] > at) & ~df["notice"]]


def run(fetch: bool = True, notify: bool = True, max_briefs: int = 15) -> dict:
    at = now_sa()
    today = at.normalize()

    tenders = load_tenders()
    if fetch:
        snap = fetch_open()
        log.info("fetched %d open tenders", len(snap))
        tenders = merge_snapshot(tenders, snap, today)

    corpus = pd.concat([load_corpus(), tenders[["description", "category", "buyer"]]],
                       ignore_index=True).drop_duplicates()
    model, report = train(corpus)
    log.info("classifier: %s examples, macro F1 %.3f", report["training_examples"],
             report["cv_macro_f1"])
    tenders = apply(model, tenders)
    tenders["notice"] = tenders["description"].fillna("").str.contains(NOTICE_RX)
    tenders = tenders.sort_values(["first_seen", "closes"], ascending=[False, True])
    save_tenders(tenders)

    briefs = load_json(BRIEFS, {})
    provider = briefs_mod.available_provider()
    budget = max_briefs if provider else 0
    live = open_tenders(tenders, at)
    digests = []
    for raw in load_json(PROFILES, []):
        profile = Profile.from_dict(raw)
        ranked = score(live, profile)
        good = ranked[ranked["score"] >= MIN_SCORE]
        new = good[good["first_seen"] >= today].head(MAX_PER_DIGEST)
        closing = good[good["closes"] <= at + timedelta(days=3)].head(5)

        for row in new.to_dict("records"):
            key = str(row["id"])
            if budget <= 0 or key in briefs:
                continue
            try:
                briefs[key] = briefs_mod.generate(row, provider)
                budget -= 1
            except Exception as exc:
                log.warning("brief for %s failed: %s", key, exc)

        sent = deliver(profile, new, closing) if notify else []
        digests.append({"profile": profile.name, "new_matches": len(new),
                        "closing_soon": len(closing), "sent": sent})
        log.info("%s: %d new, %d closing soon, sent via %s", profile.name, len(new),
                 len(closing), sent or "nothing")

    # Drop briefs for tenders that have left the archive.
    ids = set(tenders["id"].astype(str))
    save_json(BRIEFS, {k: v for k, v in briefs.items() if k in ids})

    status = {
        "updated": at.strftime("%Y-%m-%d %H:%M"),
        "open_tenders": int(len(live)),
        "new_today": int((live["first_seen"] >= today).sum()),
        "archive_size": int(len(tenders)),
        "llm_provider": provider,
        "classifier": report,
        "sector_sources": tenders["sector_source"].replace("", "unclassified")
                                                  .value_counts().to_dict(),
        "digests": digests,
    }
    save_json(STATUS, status)
    return status
