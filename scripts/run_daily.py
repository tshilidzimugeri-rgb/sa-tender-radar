"""Daily job, run by GitHub Actions on weekday mornings.

    python -m scripts.run_daily               # full run
    python -m scripts.run_daily --no-notify   # update data without sending digests
    python -m scripts.run_daily --offline     # reclassify the stored data only
"""
from __future__ import annotations

import argparse
import json
import logging

from radar.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="do not fetch new tenders")
    parser.add_argument("--no-notify", action="store_true", help="do not send digests")
    parser.add_argument("--max-briefs", type=int, default=15,
                        help="AI briefs to generate per run (keeps within free tiers)")
    args = parser.parse_args()
    status = run(fetch=not args.offline, notify=not args.no_notify,
                 max_briefs=args.max_briefs)
    print(json.dumps({k: v for k, v in status.items() if k != "classifier"}, indent=1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
