"""Cross-sectional walk-forward validation on 100 randomly sampled *real* Taiwan stocks.

The validator tests the price/benchmark core without look-ahead.  At each shared
historical cutoff date, all sampled stocks are scored using data available at or
before that date, ranked cross-sectionally, and only then are future 10/40/120-day
returns read.  This is closer to the product's real task: choose relatively better
stocks from a market at a point in time.

Current fundamentals / institutional flow are deliberately not backfilled from
present-day snapshots.  Full point-in-time accounting/flow validation requires a
separately versioned historical research store.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

from market_data import DailyPriceStore, fetch_twse_universe
from radar_service import RunSettings
from return_first_model import estimate_all_horizons

SEED = 1337
HORIZONS = {"short": 10, "mid": 40, "long": 120}
PRICE_WEIGHTS = {
    "short": (0.52, 0.20),
    "mid": (0.42, 0.22),
    "long": (0.34, 0.22),
}


def core_score(forecast: dict, horizon: str) -> float | None:
    if not forecast or not forecast.get("estimate_available"):
        return None
    try:
        technical = float(forecast.get("technical_factor_score"))
        empirical = float(forecast.get("empirical_quality_score"))
    except Exception:
        return None
    wt, we = PRICE_WEIGHTS[horizon]
    return float((technical * wt + empirical * we) / (wt + we))


def costs(settings: RunSettings) -> float:
    return settings.commission * 2 + settings.sell_tax + settings.slippage * 2


def _position_at_or_before(df: pd.DataFrame, date: pd.Timestamp) -> int | None:
    if df.empty:
        return None
    pos = int(df.index.searchsorted(pd.Timestamp(date), side="right") - 1)
    return pos if 0 <= pos < len(df) else None


def future_return_at(df: pd.DataFrame, date: pd.Timestamp, days: int, total_cost: float) -> float | None:
    pos = _position_at_or_before(df, date)
    if pos is None or pos + days >= len(df):
        return None
    a = float(df["Close"].iloc[pos])
    b = float(df["Close"].iloc[pos + days])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return None
    return b / a - 1.0 - total_cost


def benchmark_future(benchmark: pd.DataFrame, date: pd.Timestamp, days: int) -> float | None:
    pos = _position_at_or_before(benchmark, date)
    if pos is None or pos + days >= len(benchmark):
        return None
    a = float(benchmark["Close"].iloc[pos])
    b = float(benchmark["Close"].iloc[pos + days])
    return b / a - 1.0 if a > 0 else None


def choose_real_sample(
    store: DailyPriceStore,
    universe: pd.DataFrame,
    target: int,
    period: str,
    seed: int,
) -> list[str]:
    # fetch_twse_universe excludes 0-prefix ETF/ETN codes in V13.3.
    tickers = universe["ticker"].astype(str).drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(tickers)
    selected: list[str] = []
    cursor = 0
    while cursor < len(tickers) and len(selected) < target:
        batch = tickers[cursor:cursor + 80]
        cursor += len(batch)
        store.batch_fetch_and_update(batch, period=period)
        for ticker in batch:
            df = store.get_prices(ticker)
            if len(df) < 650:
                continue
            recent = df.tail(120)
            if float(recent["Volume"].fillna(0).mean()) <= 0:
                continue
            selected.append(ticker)
            if len(selected) >= target:
                break
    return selected


def shared_cutoffs(store: DailyPriceStore, sample: list[str], benchmark: pd.DataFrame, count: int) -> list[pd.Timestamp]:
    """Choose common historical dates so each stock has training history + unseen future."""
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for ticker in sample:
        df = store.get_prices(ticker)
        if len(df) < 650:
            continue
        starts.append(pd.Timestamp(df.index[380]))
        ends.append(pd.Timestamp(df.index[-121]))
    if not starts or not ends:
        return []
    lo = max(starts)
    hi = min(ends)
    eligible = benchmark[(benchmark.index >= lo) & (benchmark.index <= hi)].index
    if len(eligible) < 60:
        return []
    # Candidate dates roughly quarterly; then spread requested points across the span.
    candidates = eligible[::63]
    if len(candidates) < count:
        candidates = eligible[::42]
    if len(candidates) <= count:
        return [pd.Timestamp(x) for x in candidates]
    ids = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [pd.Timestamp(candidates[i]) for i in ids]


def _safe_spearman(g: pd.DataFrame) -> float | None:
    if len(g) < 8 or g["score"].nunique() < 3 or g["realized_net_return"].nunique() < 3:
        return None
    v = g["score"].corr(g["realized_net_return"], method="spearman")
    return None if pd.isna(v) else float(v)


def summarize(g: pd.DataFrame) -> dict:
    g = g.dropna(subset=["date", "score", "realized_net_return"]).copy()
    if g.empty:
        return {"observations": 0}

    # Rank within the same historical date. This avoids mixing bull/bear regimes
    # and directly evaluates the selector's cross-sectional purpose.
    g["score_pct_rank"] = g.groupby("date")["score"].rank(pct=True, method="average")
    top = g[g["score_pct_rank"] >= 0.80].copy()
    ics = []
    daily_lifts = []
    positive_lifts = []
    for _, d in g.groupby("date"):
        ic = _safe_spearman(d)
        if ic is not None:
            ics.append(ic)
        td = d[d["score_pct_rank"] >= 0.80]
        if not td.empty:
            daily_lifts.append(float(td["realized_net_return"].median() - d["realized_net_return"].median()))
            positive_lifts.append(float((td["realized_net_return"] > 0).mean() - (d["realized_net_return"] > 0).mean()))

    base_pos = float((g["realized_net_return"] > 0).mean())
    top_pos = float((top["realized_net_return"] > 0).mean()) if not top.empty else np.nan
    alpha_top = top["realized_alpha"].dropna()
    return {
        "observations": int(len(g)),
        "stocks": int(g["ticker"].nunique()),
        "cutoff_dates": int(g["date"].nunique()),
        "mean_cross_sectional_spearman": None if not ics else round(float(np.mean(ics)), 4),
        "median_cross_sectional_spearman": None if not ics else round(float(np.median(ics)), 4),
        "all_mean_return": round(float(g["realized_net_return"].mean()), 4),
        "all_median_return": round(float(g["realized_net_return"].median()), 4),
        "all_positive_rate": round(base_pos, 4),
        "top20_mean_return": round(float(top["realized_net_return"].mean()), 4),
        "top20_median_return": round(float(top["realized_net_return"].median()), 4),
        "top20_positive_rate": None if pd.isna(top_pos) else round(top_pos, 4),
        "top20_positive_rate_lift": None if pd.isna(top_pos) else round(top_pos - base_pos, 4),
        "mean_same_date_median_return_lift": None if not daily_lifts else round(float(np.mean(daily_lifts)), 4),
        "mean_same_date_positive_rate_lift": None if not positive_lifts else round(float(np.mean(positive_lifts)), 4),
        "top20_mean_alpha_vs_taiex": None if alpha_top.empty else round(float(alpha_top.mean()), 4),
        "top20_p10_return": round(float(top["realized_net_return"].quantile(.10)), 4),
    }


def run(
    out_dir: Path,
    target: int = 100,
    period: str = "8y",
    seed: int = SEED,
    points_per_stock: int = 6,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    store = DailyPriceStore(out_dir / "validation_prices.sqlite")
    settings = RunSettings(history_period=period, model_family="price_only")

    universe = fetch_twse_universe()
    if len(universe) < target:
        raise RuntimeError(f"股票名單僅取得 {len(universe)} 檔，無法做 {target} 檔真實標的驗證")

    store.batch_fetch_and_update(["^TWII"], period=period)
    benchmark = store.get_prices("^TWII")
    if len(benchmark) < 650:
        raise RuntimeError("TAIEX 歷史資料不足，驗證中止")

    sample = choose_real_sample(store, universe, target, period, seed)
    if len(sample) < target:
        raise RuntimeError(f"僅取得 {len(sample)} 檔具足夠歷史資料的真實標的，未達 {target} 檔")

    meta = universe.set_index("ticker").to_dict("index")
    pd.DataFrame([{"ticker": t, **meta.get(t, {})} for t in sample]).to_csv(
        out_dir / "sample_100.csv", index=False, encoding="utf-8-sig"
    )

    cutoffs = shared_cutoffs(store, sample, benchmark, points_per_stock)
    if len(cutoffs) < max(3, min(points_per_stock, 3)):
        raise RuntimeError("100 檔共同可用的歷史驗效日期不足")

    records: list[dict] = []
    total_cost = costs(settings)
    for cutoff in cutoffs:
        bench_hist = benchmark[benchmark.index <= cutoff].copy()
        if len(bench_hist) < 320:
            continue
        b20 = float(bench_hist["Close"].iloc[-1] / bench_hist["Close"].iloc[-21] - 1.0) if len(bench_hist) >= 21 else 0.0
        for ticker in sample:
            df = store.get_prices(ticker)
            stock_hist = df[df.index <= cutoff].copy()
            if len(stock_hist) < 320:
                continue
            forecasts = estimate_all_horizons(stock_hist, settings, twii_ret_20d=b20, benchmark_df=bench_hist)
            for horizon, days in HORIZONS.items():
                forecast = forecasts[horizon]
                score = core_score(forecast, horizon)
                realized = future_return_at(df, cutoff, days, total_cost)
                bret = benchmark_future(benchmark, cutoff, days)
                if score is None or realized is None:
                    continue
                records.append({
                    "ticker": ticker,
                    "date": str(cutoff.date()),
                    "horizon": horizon,
                    "days": days,
                    "score": round(score, 3),
                    "technical_score": forecast.get("technical_factor_score"),
                    "empirical_quality_score": forecast.get("empirical_quality_score"),
                    "analog_n": forecast.get("local_effective_n"),
                    "predicted_median": (forecast.get("strategy") or {}).get("median"),
                    "smoothed_positive_rate": (forecast.get("strategy") or {}).get("smoothed_positive_rate"),
                    "realized_net_return": realized,
                    "benchmark_return": bret,
                    "realized_alpha": None if bret is None else realized - bret,
                })

    obs = pd.DataFrame(records)
    if obs.empty:
        raise RuntimeError("沒有產生可用 walk-forward 觀察值")
    obs.to_csv(out_dir / "walkforward_observations.csv", index=False, encoding="utf-8-sig")

    metrics = {h: summarize(obs[obs["horizon"] == h]) for h in HORIZONS}
    report = {
        "model": "Alpha Radar V13.4 price/benchmark selection core",
        "seed": seed,
        "requested_real_stocks": target,
        "validated_real_stocks": int(obs["ticker"].nunique()),
        "period": period,
        "shared_cutoff_dates": [str(x.date()) for x in cutoffs],
        "lookahead_control": "At each cutoff, every stock and TAIEX input is truncated before future returns are read.",
        "sample_method": "Fixed-seed random draw from current TWSE/TPEx four-digit common-stock universe; only minimum history/data-quality exclusions.",
        "ranking_method": "Scores are ranked within each shared historical date; top 20% is compared with the same-date sample.",
        "scope_note": "Validates price/benchmark analog core only. Current fundamentals/flows are not backfilled into history, preventing point-in-time leakage.",
        "bias_note": "Sampling the current listed/OTC universe can retain survivorship bias because delisted historical constituents are not included.",
        "metrics": metrics,
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Alpha Radar V13.4 — 100-stock cross-sectional walk-forward validation",
        "",
        f"- Seed: `{seed}`",
        f"- Real stocks: `{report['validated_real_stocks']}` / `{target}`",
        f"- History: `{period}`",
        f"- Shared cutoff dates: `{len(cutoffs)}`",
        "- No look-ahead: each cutoff uses only information available by that date.",
        "- Ranking: top 20% is defined within the same historical date.",
        "- Scope: price/benchmark core; no present-day fundamental/flow backfill.",
        "- Limitation: current-universe sampling can retain survivorship bias.",
        "",
        "| Horizon | Obs | Mean rank IC | Top20 median | All median | Same-date median lift | Top20 positive | Base positive | Same-date positive lift | Top20 alpha | P10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        m = metrics[h]
        def pct(x):
            return "—" if x is None else f"{x*100:.2f}%"
        def dec(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append(
            f"| {h} | {m.get('observations',0)} | {dec(m.get('mean_cross_sectional_spearman'))} | "
            f"{pct(m.get('top20_median_return'))} | {pct(m.get('all_median_return'))} | "
            f"{pct(m.get('mean_same_date_median_return_lift'))} | {pct(m.get('top20_positive_rate'))} | "
            f"{pct(m.get('all_positive_rate'))} | {pct(m.get('mean_same_date_positive_rate_lift'))} | "
            f"{pct(m.get('top20_mean_alpha_vs_taiex'))} | {pct(m.get('top20_p10_return'))} |"
        )
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/validation_v134")
    ap.add_argument("--stocks", type=int, default=100)
    ap.add_argument("--period", default="8y")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--points", type=int, default=6, help="shared historical cutoff dates")
    args = ap.parse_args()
    try:
        report = run(Path(args.out), args.stocks, args.period, args.seed, args.points)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "NOT_RUN",
            "reason": f"{type(exc).__name__}: {exc}",
            "note": "No synthetic substitution was used. Real-stock validation requires live historical-data access.",
        }
        (out / "validation_not_run.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
