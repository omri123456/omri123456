"""Command line: python -m marketlab <command>.

  fetch      download / refresh every data source into data/raw
  build      write the feature panel to data/processed (for inspection)
  research   event study and feature information coefficients -> runs/research
  train      walk-forward training of all model variants -> runs/latest, models/production.pt
  backtest   out-of-sample scoring and virtual-money simulation -> runs/latest
  paper      forward paper account: init | step | status
  report     build reports/latest.html
  daily      fetch -> paper step -> breadth scan -> signal -> report  (every trading day)
  signal     print today's decision again from what is on disk
  dashboard  write the decision dashboard's data files to reports/dashboard
  run-all    fetch -> research -> train -> backtest -> paper -> report
  status     what is on disk and how fresh it is
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import pipeline
from .config import Paths, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marketlab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML file layered over config/default.yaml")
    parser.add_argument("--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "build", "research", "backtest", "report", "run-all", "status", "signal", "dashboard"):
        sub.add_parser(name)
    daily = sub.add_parser("daily")
    daily.add_argument("--no-scan", action="store_true", help="skip the market-breadth scan")
    train = sub.add_parser("train")
    train.add_argument("--variants", help="comma separated, default: all (" + ", ".join(pipeline.VARIANTS) + ")")
    train.add_argument("--years", help="comma separated test years to (re)train, default: all")
    paper = sub.add_parser("paper")
    paper.add_argument("action", choices=["init", "step", "status"])
    paper.add_argument("--start", help="init only: open the account on this past date (YYYY-MM-DD)")
    paper.add_argument("--force", action="store_true", help="init only: replace an existing account")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    paths = Paths().ensure()

    def report() -> None:
        from .report import build
        path = build.build_report(cfg, paths)
        logging.getLogger(__name__).info("report written to %s", path)

    if args.command == "fetch":
        pipeline.do_fetch(cfg, paths)
    elif args.command == "build":
        pipeline.do_build(cfg, paths)
    elif args.command == "research":
        pipeline.do_research(cfg, paths)
    elif args.command == "train":
        variants = args.variants.split(",") if args.variants else None
        years = [int(y) for y in args.years.split(",")] if args.years else None
        pipeline.do_train(cfg, paths, variants, years)
    elif args.command == "backtest":
        pipeline.do_backtest(cfg, paths)
    elif args.command == "paper":
        info = pipeline.do_paper(cfg, paths, args.action, args.start, args.force)
        print(json.dumps(pipeline.as_jsonable(info), indent=1))
    elif args.command == "report":
        report()
    elif args.command == "daily":
        from .trading.signal import describe
        pipeline.do_fetch(cfg, paths)
        pipeline.do_paper(cfg, paths, "step")
        result = pipeline.do_signal(cfg, paths, scan=not args.no_scan)
        report()
        try:
            pipeline.do_dashboard(cfg, paths)
        except Exception as exc:  # the dashboard is a view: never block the daily signal on it
            logging.getLogger(__name__).warning("dashboard data not built: %s", exc)
        print(describe(result))
    elif args.command == "dashboard":
        for name, path in pipeline.do_dashboard(cfg, paths).items():
            print(name, path)
    elif args.command == "signal":
        from .trading.signal import describe
        print(describe(pipeline.do_signal(cfg, paths, scan=False)))
    elif args.command == "run-all":
        pipeline.do_fetch(cfg, paths)
        pipeline.do_research(cfg, paths)
        pipeline.do_train(cfg, paths)
        pipeline.do_backtest(cfg, paths)
        pipeline.do_paper(cfg, paths, "step")
        report()
    elif args.command == "status":
        print(json.dumps(pipeline.as_jsonable(pipeline.do_status(cfg, paths)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
