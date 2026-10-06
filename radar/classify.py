"""Sector classifier trained by weak supervision.

The buyer-selected category on eTenders mixes two vocabularies and is often generic
("Services: General"), so it cannot be used directly. Instead, labels come from the
keyword rules in `radar.taxonomy`, applied to the tender description. Those rules are
precise but only cover part of the data; a linear model trained on the rule-labelled
tenders labels the rest.

The model sees the description with the rule keywords removed, plus the buyer and the
buyer-selected category. Tenders the rules could not label contain no keywords, so this
makes training look like the data the model is used on, and it makes the cross-validated
score an honest estimate of accuracy on those tenders instead of a measure of how well it
re-learns the rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from radar.taxonomy import OTHER, mask_rules, rule_label, tied_sectors

MIN_CONFIDENCE = 0.4
MIN_EXAMPLES = 30


def model_text(df: pd.DataFrame) -> pd.Series:
    desc = df["description"].fillna("").str.lower().map(mask_rules)
    return (desc + " | " + df["category"].fillna("").str.lower()
            + " | " + df["buyer"].fillna("").str.lower())


def _pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                                  sublinear_tf=True, stop_words="english",
                                  token_pattern=r"(?u)\b[a-z][a-z]+\b")),
        ("clf", LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced")),
    ])


def train(corpus: pd.DataFrame, evaluate: bool = True) -> tuple[Pipeline, dict]:
    """Fit on every rule-labelled tender in `corpus`. Returns the model and a report."""
    labels = corpus["description"].fillna("").map(rule_label)
    counts = labels.value_counts()
    keep = labels.isin(counts[counts >= MIN_EXAMPLES].index)
    x, y = model_text(corpus[keep]), labels[keep]

    report = {"training_examples": int(keep.sum()), "corpus_size": len(corpus),
              "classes": sorted(y.unique())}
    if evaluate:
        folds = StratifiedKFold(5, shuffle=True, random_state=0)
        pred = cross_val_predict(_pipeline(), x, y, cv=folds)
        cr = classification_report(y, pred, output_dict=True, zero_division=0)
        report["cv_accuracy"] = round(cr["accuracy"], 3)
        report["cv_macro_f1"] = round(cr["macro avg"]["f1-score"], 3)
        report["per_sector_f1"] = {s: round(cr[s]["f1-score"], 3) for s in report["classes"]}
    return _pipeline().fit(x, y), report


def apply(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Add sector, sector_2, sector_source and sector_confidence columns."""
    df = df.copy()
    classes = np.array(model.classes_)
    proba = model.predict_proba(model_text(df))

    rules = df["description"].fillna("").map(rule_label)
    for i, desc in enumerate(df["description"].fillna("")):
        # Where the rules tie between sectors, let the model choose among those only.
        tied = [j for j, c in enumerate(classes) if c in tied_sectors(desc)]
        if tied and proba[i, tied].sum() > 0:
            row = np.zeros(len(classes))
            row[tied] = proba[i, tied] / proba[i, tied].sum()
            proba[i] = row

    order = np.argsort(-proba, axis=1)
    best, second = classes[order[:, 0]], classes[order[:, 1]]
    conf = proba[np.arange(len(df)), order[:, 0]]
    model_sector = np.where(conf >= MIN_CONFIDENCE, best, OTHER)

    from_rules = rules.notna().to_numpy()
    df["sector"] = np.where(from_rules, rules.fillna(""), model_sector)
    df["sector_source"] = np.where(from_rules, "rules",
                                   np.where(model_sector == OTHER, "", "model"))
    df["sector_confidence"] = np.where(from_rules, 1.0, conf).round(3)
    # The model's next-best sector, used for secondary matches.
    df["sector_2"] = np.where(best == df["sector"], second, best)
    return df
