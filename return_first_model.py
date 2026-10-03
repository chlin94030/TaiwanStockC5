"""Taiwan Alpha Radar V13.3 - walk-forward safe return / technical engine.

Key rules
---------
* Rank-first: usable stocks receive a relative score; missing evidence does not
  silently become a zero.
* No fabricated forecast: insufficient history returns estimate_available=False.
* Historical analogs only use rows whose outcomes are already known inside the
  supplied slice. In walk-forward validation, callers pass data truncated at the
  evaluation date, so there is no look-ahead.
* Relative strength is aligned to the benchmark at EACH historical date instead
  of subtracting today's benchmark return from every historical observation.
* Neighboring analog dates are de-overlapped so one market swing cannot masquerade
  as dozens of independent examples.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


class ModelDataError(Exception):
    pass


def finite_scalar(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev = close.shift(1)
    return pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def _benchmark_frame(index: pd.Index, benchmark_df: pd.DataFrame | None, fallback_ret20: float) -> pd.DataFrame:
    """Return benchmark features aligned to stock dates.

    When a benchmark is unavailable we keep a conservative fallback for the
    current scan, but the walk-forward validator always supplies benchmark data.
    """
    out = pd.DataFrame(index=index)
    if benchmark_df is None or benchmark_df.empty or "Close" not in benchmark_df.columns:
        out["market_ret20"] = float(fallback_ret20 or 0.0)
        out["market_gap20"] = 0.0
        out["market_slope20"] = 0.0
        return out

    b = benchmark_df[["Close"]].copy().sort_index()
    b.index = pd.to_datetime(b.index).tz_localize(None)
    close = pd.to_numeric(b["Close"], errors="coerce")
    ma20 = close.rolling(20).mean()
    feat = pd.DataFrame(index=b.index)
    feat["market_ret20"] = close.pct_change(20)
    feat["market_gap20"] = close / ma20 - 1.0
    feat["market_slope20"] = ma20.pct_change(10)
    # Taiwan stock/index dates normally match. ffill only bridges rare missing index rows.
    feat = feat.reindex(pd.to_datetime(index)).ffill(limit=2)
    feat.index = index
    feat["market_ret20"] = feat["market_ret20"].fillna(float(fallback_ret20 or 0.0))
    feat[["market_gap20", "market_slope20"]] = feat[["market_gap20", "market_slope20"]].fillna(0.0)
    return feat


def _feature_frame(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    fallback_market_ret20: float = 0.0,
) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    x = df[["Open", "High", "Low", "Close", "Volume"]].copy().sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    for c in required:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Close"])
    if x.empty:
        return pd.DataFrame()

    close = x["Close"]
    vol = x["Volume"].fillna(0.0)
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    ma240 = close.rolling(240).mean()

    atr14 = _true_range(x).rolling(14).mean()
    vol20 = vol.rolling(20).mean()
    high20_prev = x["High"].shift(1).rolling(20).max()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)

    # KD (9,3,3) kept in the model even though it is hidden from the default
    # mobile chart.  It is used as a light entry-timing signal, not a hard gate.
    low9 = x["Low"].rolling(9).min()
    high9 = x["High"].rolling(9).max()
    rsv = ((close - low9) / (high9 - low9).replace(0, np.nan) * 100.0).clip(0, 100)
    kd_k = rsv.ewm(com=2, adjust=False).mean()
    kd_d = kd_k.ewm(com=2, adjust=False).mean()

    f = pd.DataFrame(index=x.index)
    f["ret5"] = close.pct_change(5)
    f["ret20"] = close.pct_change(20)
    f["ret60"] = close.pct_change(60)
    f["ret120"] = close.pct_change(120)
    f["gap20"] = close / ma20 - 1.0
    f["gap60"] = close / ma60 - 1.0
    f["gap120"] = close / ma120 - 1.0
    f["gap240"] = close / ma240 - 1.0
    f["ma20_slope"] = ma20.pct_change(10)
    f["ma60_slope"] = ma60.pct_change(20)
    f["vol_ratio"] = vol / vol20.replace(0, np.nan)
    f["atr_pct"] = atr14 / close.replace(0, np.nan)
    f["rsi14"] = _rsi(close) / 100.0
    f["macd_pct"] = macd_hist / close.replace(0, np.nan)
    f["kd_k"] = kd_k / 100.0
    f["kd_d"] = kd_d / 100.0
    f["kd_spread"] = (kd_k - kd_d) / 100.0
    f["bb_pos"] = ((close - bb_lower) / bb_width).clip(-0.5, 1.5)
    f["breakout20"] = (close / high20_prev.replace(0, np.nan) - 1.0).clip(-0.30, 0.30)

    daily = close.pct_change()
    f["volatility20"] = daily.rolling(20).std()
    f["drawdown60"] = close / close.rolling(60).max() - 1.0

    market = _benchmark_frame(f.index, benchmark_df, fallback_market_ret20)
    f = f.join(market)
    f["rs20"] = f["ret20"] - f["market_ret20"]

    # Diagnostic-only raw series.
    f["close"] = close
    f["ma5"] = ma5
    f["ma20"] = ma20
    f["ma60"] = ma60
    f["ma120"] = ma120
    f["ma240"] = ma240
    f["atr14"] = atr14
    return f.replace([np.inf, -np.inf], np.nan)


def _technical_score(row: pd.Series, horizon: str) -> float:
    """Bounded 0-100 technical quality score; no single hard gate."""
    r5 = finite_scalar(row.get("ret5"))
    r20 = finite_scalar(row.get("ret20"))
    r60 = finite_scalar(row.get("ret60"))
    r120 = finite_scalar(row.get("ret120"))
    rs20 = finite_scalar(row.get("rs20"))
    vr = finite_scalar(row.get("vol_ratio"), 1.0)
    g20 = finite_scalar(row.get("gap20"))
    g60 = finite_scalar(row.get("gap60"))
    g120 = finite_scalar(row.get("gap120"))
    slope20 = finite_scalar(row.get("ma20_slope"))
    slope60 = finite_scalar(row.get("ma60_slope"))
    rsi = finite_scalar(row.get("rsi14"), 0.5)
    macd = finite_scalar(row.get("macd_pct"))
    kd_k = finite_scalar(row.get("kd_k"), 0.5)
    kd_spread = finite_scalar(row.get("kd_spread"))
    breakout = finite_scalar(row.get("breakout20"))
    atr = finite_scalar(row.get("atr_pct"), 0.03)
    dd60 = finite_scalar(row.get("drawdown60"), -0.05)
    market_gap = finite_scalar(row.get("market_gap20"))

    def s(v, lo, hi):
        return float(_clip01((v - lo) / max(1e-9, hi - lo)))

    trend = (
        0.28 * s(g20, -0.05, 0.08)
        + 0.24 * s(g60, -0.08, 0.16)
        + 0.20 * s(g120, -0.12, 0.25)
        + 0.16 * s(slope20, -0.03, 0.08)
        + 0.12 * s(slope60, -0.05, 0.12)
    )
    relative = s(rs20, -0.08, 0.12)
    volume = s(vr, 0.65, 1.80)
    breakout_s = s(breakout, -0.06, 0.035)
    macd_s = s(macd, -0.01, 0.015)
    kd_cross_s = s(kd_spread, -0.12, 0.12)
    # Prefer constructive KD (roughly 35-80) over deeply weak or extremely hot levels.
    kd_level_s = max(0.0, 1.0 - abs(kd_k - 0.58) / 0.48)
    kd_s = 0.65 * kd_cross_s + 0.35 * kd_level_s
    momentum_short = 0.45 * s(r5, -0.06, 0.10) + 0.55 * s(r20, -0.10, 0.22)
    momentum_mid = 0.35 * s(r20, -0.12, 0.25) + 0.65 * s(r60, -0.18, 0.45)
    momentum_long = 0.25 * s(r60, -0.22, 0.55) + 0.75 * s(r120, -0.30, 0.80)

    # Constructive but not overbought RSI, plus volatility/drawdown quality.
    rsi_quality = max(0.0, 1.0 - abs(rsi - 0.60) / 0.35)
    risk_quality = 0.65 * (1.0 - s(atr, 0.02, 0.085)) + 0.35 * s(dd60, -0.30, -0.02)
    # Mild market-context term; stock-relative strength remains the main market adjustment.
    market_quality = s(market_gap, -0.08, 0.08)

    if horizon == "short":
        # V13.4: reward sustained strength more than a one- or two-day sprint.
        # Earlier builds could let recent momentum + volume dominate the list,
        # which was useful for finding hot stocks but too prone to late entries.
        score = (
            0.16 * momentum_short + 0.20 * relative + 0.10 * volume + 0.10 * breakout_s
            + 0.16 * trend + 0.08 * macd_s + 0.06 * kd_s + 0.04 * rsi_quality + 0.07 * risk_quality + 0.03 * market_quality
        )
    elif horizon == "mid":
        score = (
            0.23 * trend + 0.19 * momentum_mid + 0.18 * relative + 0.09 * volume
            + 0.08 * breakout_s + 0.07 * macd_s + 0.05 * kd_s + 0.05 * rsi_quality + 0.04 * risk_quality + 0.02 * market_quality
        )
    else:
        score = (
            0.34 * trend + 0.22 * momentum_long + 0.15 * relative + 0.05 * volume
            + 0.04 * breakout_s + 0.05 * macd_s + 0.03 * kd_s + 0.04 * rsi_quality + 0.06 * risk_quality + 0.02 * market_quality
        )

    # V13.3.2: explicit anti-chasing penalty.  The original Stage-4/Stage-5
    # concept identified >12% / >20% distance from MA20 as extended, but that
    # label did not feed back into ranking.  Momentum could therefore dominate
    # the short list after several consecutive up days.  Keep rank-first (no
    # hard gate), but lower the rank smoothly once price becomes crowded.
    gap_pen = 0.0
    if g20 > 0.12:
        gap_pen = 4.0 + min(8.0, max(0.0, (g20 - 0.12) / 0.12 * 8.0))
    if g20 > 0.20:
        gap_pen += min(4.0, max(0.0, (g20 - 0.20) / 0.12 * 4.0))
    r5_pen = min(5.0, max(0.0, (r5 - 0.10) / 0.12 * 5.0))
    rsi_pen = min(4.0, max(0.0, (rsi - 0.78) / 0.16 * 4.0))
    if horizon == "short":
        overheat_penalty = gap_pen + r5_pen + rsi_pen
    elif horizon == "mid":
        overheat_penalty = 0.45 * gap_pen + 0.35 * r5_pen + 0.25 * rsi_pen
    else:
        overheat_penalty = 0.15 * gap_pen + 0.10 * r5_pen

    return round(float(np.clip(score * 100.0 - overheat_penalty, 0.0, 100.0)), 1)


def _deoverlap_nearest(dist: pd.Series, horizon_days: int, target_n: int) -> list:
    """Greedy nearest-neighbor selection with calendar spacing.

    Overlapping forward-return windows are highly correlated. We first require a
    spacing of roughly one quarter of the holding horizon, then relax once if the
    dataset is small. This is not independence, but materially reduces duplicate
    evidence versus taking adjacent dates as separate analogs.
    """
    ordered = list(dist.sort_values().index)
    for sep in [max(3, horizon_days // 4), max(2, horizon_days // 8), 1]:
        chosen: list = []
        for idx in ordered:
            ts = pd.Timestamp(idx)
            if all(abs((ts - pd.Timestamp(c)).days) >= sep for c in chosen):
                chosen.append(idx)
                if len(chosen) >= target_n:
                    return chosen
        if len(chosen) >= 26:
            return chosen
    return chosen


def _historical_analog_returns(
    features: pd.DataFrame,
    horizon_days: int,
    total_cost: float,
    current_row: pd.Series,
) -> dict:
    cols = [
        "ret5", "ret20", "ret60", "gap20", "gap60", "ma20_slope", "vol_ratio",
        "atr_pct", "rsi14", "macd_pct", "kd_spread", "bb_pos", "breakout20", "drawdown60",
        "rs20", "market_ret20", "market_gap20",
    ]
    hist = features[cols + ["close"]].copy()
    hist["fwd"] = hist["close"].shift(-horizon_days) / hist["close"] - 1.0 - total_cost
    # At an evaluation date, the final horizon_days outcomes are not yet known.
    hist = hist.iloc[:-max(1, horizon_days)].dropna(subset=cols + ["fwd"])
    if len(hist) < 55:
        return {"available": False, "n": len(hist), "reason": "HISTORY_TOO_SHORT"}

    cur = pd.to_numeric(current_row.reindex(cols), errors="coerce")
    if cur.isna().any():
        return {"available": False, "n": len(hist), "reason": "CURRENT_FEATURE_MISSING"}

    X = hist[cols].astype(float)
    med = X.median()
    mad = (X - med).abs().median().replace(0, np.nan)
    std = X.std(ddof=0).replace(0, np.nan)
    scale = (mad * 1.4826).fillna(std).fillna(1.0).clip(lower=1e-6)
    z_hist = (X - med) / scale
    z_cur = (cur - med) / scale

    weights = pd.Series(
        [1.0, 1.25, 0.85, 1.15, 0.95, 0.8, 0.65, 0.75, 0.5, 0.5, 0.35, 0.4, 0.8, 0.65, 1.25, 0.85, 0.65],
        index=cols,
    )
    dist = np.sqrt(((z_hist - z_cur) ** 2 * weights).sum(axis=1) / weights.sum())
    target_n = int(np.clip(round(len(hist) * 0.10), 36, 100))
    nearest_idx = _deoverlap_nearest(dist, horizon_days, target_n)
    sample = hist.loc[nearest_idx, "fwd"].replace([np.inf, -np.inf], np.nan).dropna()
    nearest_dist = dist.loc[sample.index].dropna()
    if len(sample) < 26:
        return {"available": False, "n": len(sample), "reason": "ANALOG_SAMPLE_TOO_SMALL"}

    q10 = float(sample.quantile(0.10))
    lower = sample[sample <= q10]
    es10 = float(lower.mean()) if not lower.empty else q10
    mean = float(sample.mean())
    median = float(sample.median())
    p75 = float(sample.quantile(0.75))
    wins = int((sample > 0).sum())
    raw_win_rate = wins / len(sample)
    # Beta(2,2) shrinkage keeps small-sample win rates from looking falsely precise.
    smoothed_win_rate = (wins + 2.0) / (len(sample) + 4.0)
    dispersion = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0

    n_score = min(1.0, len(sample) / 70.0)
    similarity = float(np.exp(-max(0.0, float(nearest_dist.median())) / 2.0)) if not nearest_dist.empty else 0.0
    sign_strength = min(1.0, abs(smoothed_win_rate - 0.5) * 2.0)
    stability = 1.0 - min(1.0, dispersion / max(0.04, abs(median) * 4.0 + 0.04))
    evidence_score = 100.0 * (0.34 * n_score + 0.38 * similarity + 0.16 * sign_strength + 0.12 * stability)

    # Empirical quality is a ranking input, not a probability forecast.
    empirical_quality = (
        50.0
        + 24.0 * np.tanh(median * 8.0)
        + 12.0 * np.tanh(mean * 6.0)
        + 14.0 * np.tanh((smoothed_win_rate - 0.5) * 4.0)
        - 35.0 * max(0.0, -q10 - 0.06)
    )

    return {
        "available": True,
        "n": int(len(sample)),
        "mean": mean,
        "median": median,
        "p75": p75,
        "p10": q10,
        "expected_shortfall10_loss": es10,
        "win_rate": raw_win_rate,
        "smoothed_win_rate": smoothed_win_rate,
        "dispersion": dispersion,
        "median_distance": float(nearest_dist.median()) if not nearest_dist.empty else math.nan,
        "evidence_score": float(np.clip(evidence_score, 15.0, 98.0)),
        "empirical_quality": float(np.clip(empirical_quality, 0.0, 100.0)),
    }


def _estimate_from_features(
    features: pd.DataFrame,
    history_len: int,
    horizon: str,
    settings,
    twii_ret_20d: float = 0.005,
) -> dict:
    min_history = {"short": 180, "mid": 220, "long": 320}.get(horizon, 220)
    if history_len < min_history:
        return {"estimate_available": False, "sample_supported": False, "reason": "PRICE_HISTORY_TOO_SHORT"}
    if features is None or features.empty:
        return {"estimate_available": False, "sample_supported": False, "reason": "FEATURES_UNAVAILABLE"}

    horizon_days = {"short": 10, "mid": 40, "long": 120}.get(horizon, 40)
    current = features.iloc[-1]
    factor_score = _technical_score(current, horizon)
    total_cost = (
        finite_scalar(getattr(settings, "commission", 0.001425)) * 2.0
        + finite_scalar(getattr(settings, "sell_tax", 0.003))
        + finite_scalar(getattr(settings, "slippage", 0.0005)) * 2.0
    )
    analog = _historical_analog_returns(features, horizon_days, total_cost, current)

    rs20 = finite_scalar(current.get("rs20"))
    market_ret20 = finite_scalar(current.get("market_ret20"), twii_ret_20d)
    base = {
        "estimate_available": bool(analog.get("available", False)),
        "sample_supported": bool(analog.get("available", False)),
        "is_outperforming_market": rs20 > 0,
        "composite_factor_score": factor_score,
        "technical_factor_score": factor_score,
        "alpha_mean": round(rs20, 4),
        "market_ret20": round(market_ret20, 4),
        "features": {
            "ret5": round(finite_scalar(current.get("ret5")), 4),
            "ret20": round(finite_scalar(current.get("ret20")), 4),
            "ret60": round(finite_scalar(current.get("ret60")), 4),
            "vol_ratio": round(finite_scalar(current.get("vol_ratio"), 1.0), 3),
            "atr_pct": round(finite_scalar(current.get("atr_pct")), 4),
            "rsi14": round(finite_scalar(current.get("rsi14"), 0.5) * 100.0, 1),
            "kd_k": round(finite_scalar(current.get("kd_k"), 0.5) * 100.0, 1),
            "kd_d": round(finite_scalar(current.get("kd_d"), 0.5) * 100.0, 1),
            "kd_spread": round(finite_scalar(current.get("kd_spread")), 4),
            "macd_pct": round(finite_scalar(current.get("macd_pct")), 5),
            "gap20": round(finite_scalar(current.get("gap20")), 4),
            "gap60": round(finite_scalar(current.get("gap60")), 4),
            "market_gap20": round(finite_scalar(current.get("market_gap20")), 4),
        },
    }
    if not analog.get("available"):
        base.update({
            "confidence_score": 0.0,
            "empirical_quality_score": None,
            "strategy": {},
            "local_effective_n": float(analog.get("n", 0)),
            "local_time_blocks": 0,
            "local_weight": 0.0,
            "reason": analog.get("reason", "ANALOG_SAMPLE_TOO_SMALL"),
        })
        return base

    base.update({
        "confidence_score": round(finite_scalar(analog.get("evidence_score")), 1),
        "empirical_quality_score": round(finite_scalar(analog.get("empirical_quality")), 1),
        "strategy": {
            "mean": round(finite_scalar(analog.get("mean")), 4),
            "median": round(finite_scalar(analog.get("median")), 4),
            "p75": round(finite_scalar(analog.get("p75")), 4),
            "p10": round(finite_scalar(analog.get("p10")), 4),
            "expected_shortfall10_loss": round(finite_scalar(analog.get("expected_shortfall10_loss")), 4),
            "historical_positive_rate": round(finite_scalar(analog.get("win_rate")), 4),
            "smoothed_positive_rate": round(finite_scalar(analog.get("smoothed_win_rate")), 4),
            "dispersion": round(finite_scalar(analog.get("dispersion")), 4),
        },
        "local_effective_n": float(analog.get("n", 0)),
        "local_time_blocks": int(max(1, analog.get("n", 0) // 15)),
        "local_weight": round(float(np.clip(analog.get("n", 0) / 75.0, 0.30, 0.95)), 2),
        "analog_median_distance": round(finite_scalar(analog.get("median_distance")), 3),
    })
    return base


def estimate_horizon_return(
    df: pd.DataFrame,
    horizon: str,
    settings,
    twii_ret_20d: float = 0.005,
    benchmark_df: pd.DataFrame | None = None,
) -> dict:
    if df is None or df.empty:
        return {"estimate_available": False, "sample_supported": False, "reason": "PRICE_HISTORY_TOO_SHORT"}
    features = _feature_frame(df, benchmark_df=benchmark_df, fallback_market_ret20=twii_ret_20d)
    return _estimate_from_features(features, len(df), horizon, settings, twii_ret_20d)


def estimate_all_horizons(
    df: pd.DataFrame,
    settings,
    twii_ret_20d: float = 0.005,
    benchmark_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Compute the expensive feature matrix once, then evaluate all horizons."""
    if df is None or df.empty:
        empty = {"estimate_available": False, "sample_supported": False, "reason": "PRICE_HISTORY_TOO_SHORT"}
        return {h: dict(empty) for h in ["short", "mid", "long"]}
    features = _feature_frame(df, benchmark_df=benchmark_df, fallback_market_ret20=twii_ret_20d)
    return {
        h: _estimate_from_features(features, len(df), h, settings, twii_ret_20d)
        for h in ["short", "mid", "long"]
    }


def holding_review(price: float, original_invalidation=None, trailing_protection=None, thesis_broken=None, prices_verified=True) -> str:
    if not prices_verified:
        return "DATA_UNVERIFIED"
    if original_invalidation and price <= original_invalidation:
        return "ORIGINAL_STRUCTURE_INVALIDATED"
    if thesis_broken is True:
        return "ORIGINAL_THESIS_INVALIDATED"
    if trailing_protection and price <= trailing_protection:
        return "PROTECTION_TRIGGER_REVIEW_EXECUTION"
    if not original_invalidation and not trailing_protection:
        return "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW"
    return "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE"
