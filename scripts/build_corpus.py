"""Build the classifier training corpus from cancelled eTenders listings.

Cancelled tenders are a free, large (25,000+) and varied sample of real tender
descriptions. Only the text fields are kept. Run occasionally:

    python -m scripts.build_corpus
"""
from __future__ import annotations

import logging

from radar.etenders import CANCELLED, fetch_raw, to_frame
from radar.store import CORPUS, DATA


def main() -> None:
    df = to_frame(fetch_raw(CANCELLED, timeout=900))
    corpus = (df[["description", "category", "buyer"]]
              .query("description != ''")
              .drop_duplicates())
    DATA.mkdir(exist_ok=True)
    corpus.to_csv(CORPUS, index=False, compression={"method": "gzip", "mtime": 0})
    logging.info("saved %d rows to %s", len(corpus), CORPUS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
