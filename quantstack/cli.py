from __future__ import annotations
import argparse
import sys
from typing import List

from .data.fred import fetch_series, cache_series
from .data.yahoo import fetch_prices, cache_prices
from .features.rates_curve import compute_ten_two_spread
from .strategies import micro10y, wti_trend, curve_10_2, us_uk_dislocation
from .models.nowcast_macro import build_fx_nowcast


def cmd_fetch(args: argparse.Namespace) -> None:
    if args.source == "fred":
        path = cache_series(args.series, args.cache or "fred")
        print(f"Saved to {path}")
    elif args.source == "yahoo":
        path = cache_prices(args.tickers, args.cache or "yahoo")
        print(f"Saved to {path}")
    else:
        print("Unknown source", file=sys.stderr)
        sys.exit(1)


def cmd_features(args: argparse.Namespace) -> None:
    fred = fetch_series(["DGS10", "DGS2"]).ffill()
    spread = compute_ten_two_spread(fred)
    print(spread.tail().to_string())


def cmd_run_strategy(args: argparse.Namespace) -> None:
    strategies = {
        "micro10y": micro10y.run,
        "wti": wti_trend.run,
        "curve10_2": curve_10_2.run,
        "us_uk_fx": us_uk_dislocation.run,
    }
    if args.name not in strategies:
        print("Choose from:", list(strategies.keys()))
        sys.exit(1)
    stats, summ = strategies[args.name]()
    print(summ)
    print(stats.tail().to_string())


def cmd_nowcast(args: argparse.Namespace) -> None:
    fred = fetch_series(["DGS10", "DGS2"]).ffill()
    ten_two = compute_ten_two_spread(fred)
    fx = fetch_prices(["GBPUSD=X"])  # FX
    aligned = fx[["GBPUSD=X"]].join(ten_two, how="outer").ffill()
    nowcast = build_fx_nowcast(aligned.rename(columns={"GBPUSD=X": "FX"}))
    print(nowcast.tail().to_string())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantstack", description="Quant Trading Stack CLI")
    sub = p.add_subparsers(dest="cmd")

    p_fetch = sub.add_parser("fetch", help="Fetch and cache data")
    p_fetch.add_argument("source", choices=["fred", "yahoo"]) 
    p_fetch.add_argument("--series", nargs="*", default=[], help="FRED series ids")
    p_fetch.add_argument("--tickers", nargs="*", default=[], help="Yahoo tickers")
    p_fetch.add_argument("--cache", default=None, help="Cache name")
    p_fetch.set_defaults(func=cmd_fetch)

    p_feat = sub.add_parser("features", help="Build and print basic features")
    p_feat.add_argument("action", choices=["build"], nargs="?")
    p_feat.set_defaults(func=cmd_features)

    p_strat = sub.add_parser("run-strategy", help="Run a strategy backtest")
    p_strat.add_argument("name", choices=["micro10y", "wti", "curve10_2", "us_uk_fx"]) 
    p_strat.set_defaults(func=cmd_run_strategy)

    p_now = sub.add_parser("nowcast", help="Run FX nowcast demo")
    p_now.set_defaults(func=cmd_nowcast)

    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()