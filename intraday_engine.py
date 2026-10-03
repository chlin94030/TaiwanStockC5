"""Intraday overlay for Taiwan Alpha Radar V13.3.

The daily model decides *what is worth watching*.  This module only answers
*what is happening now* and re-orders that already-vetted pool.  It never
replaces the long-horizon evidence with a few minutes of price action.

Realtime source: FinMind Sponsor tick snapshot.  No scraping fallback is used;
when realtime entitlement is unavailable the caller receives an explicit
unavailable state rather than fabricated/stale intraday data.
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import requests

REALTIME_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_tick_snapshot"
INTRADAY_BLEND = {"short": 0.45, "mid": 0.15, "long": 0.05}


def _taipei_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))


def _stock_id(ticker: str) -> str:
    return str(ticker or "").split(".")[0].strip()


def market_is_open(now: datetime.datetime | None = None) -> bool:
    now = now or _taipei_now()
    if now.weekday() >= 5:
        return False
    t = now.timetz().replace(tzinfo=None)
    return datetime.time(9, 0) <= t <= datetime.time(13, 30)


def _finite(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _clip01(v: float, lo: float, hi: float) -> float:
    if not np.isfinite(v):
        return 0.5
    if hi <= lo:
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class RealtimeStatus:
    available: bool
    reason: str = ""
    quote_date: str = ""
    source: str = "FinMind realtime snapshot"
    market_open: bool = False


class FinMindRealtimeClient:
    def __init__(self, token: str | None = None):
        self.token = (token if token is not None else os.getenv("FINMIND_TOKEN", "")).strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 AlphaRadar/13.3"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    @property
    def configured(self) -> bool:
        return bool(self.token)

    def _request_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        params: list[tuple[str, str]] = []
        for stock_id in ids:
            params.append(("data_id", stock_id))
        # Keep the token in both header and query for compatibility with FinMind
        # deployments documented across versions.
        if self.token:
            params.append(("token", self.token))
        r = self.session.get(REALTIME_URL, params=params, timeout=12)
        r.raise_for_status()
        js = r.json()
        if not isinstance(js, dict):
            return []
        status = js.get("status")
        if status not in (None, 200, "200"):
            msg = str(js.get("msg") or js.get("message") or "FINMIND_API_ERROR")
            raise RuntimeError(msg)
        data = js.get("data", [])
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict)]

    def snapshots(self, tickers: Iterable[str], batch_size: int = 45) -> tuple[pd.DataFrame, RealtimeStatus]:
        if not self.configured:
            return pd.DataFrame(), RealtimeStatus(False, "FINMIND_TOKEN_NOT_CONFIGURED", market_open=market_is_open())
        ids = list(dict.fromkeys([_stock_id(t) for t in tickers if _stock_id(t)]))
        # 001 = TAIEX, 101 = OTC index in FinMind realtime endpoint.
        ids_with_index = ids + ["001", "101"]
        rows: list[dict] = []
        try:
            for i in range(0, len(ids_with_index), batch_size):
                rows.extend(self._request_ids(ids_with_index[i:i + batch_size]))
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "")
            return pd.DataFrame(), RealtimeStatus(False, f"HTTP_{status or 'ERROR'}", market_open=market_is_open())
        except Exception as exc:
            return pd.DataFrame(), RealtimeStatus(False, f"{type(exc).__name__}", market_open=market_is_open())

        if not rows:
            return pd.DataFrame(), RealtimeStatus(False, "NO_REALTIME_ROWS", market_open=market_is_open())
        df = pd.DataFrame(rows)
        if "stock_id" not in df.columns:
            return pd.DataFrame(), RealtimeStatus(False, "MISSING_STOCK_ID", market_open=market_is_open())
        df["stock_id"] = df["stock_id"].astype(str)
        if "date" in df.columns:
            # FinMind may return timestamp or date. Preserve exact source text but
            # derive the newest YYYY-MM-DD safely.
            ds = pd.to_datetime(df["date"], errors="coerce")
            quote_date = str(ds.max().date()) if ds.notna().any() else ""
        else:
            quote_date = ""
        open_now = market_is_open()
        today = _taipei_now().date().isoformat()
        if open_now and quote_date and quote_date != today:
            return pd.DataFrame(), RealtimeStatus(
                False, f"STALE_REALTIME_{quote_date}", quote_date=quote_date, market_open=True
            )
        return df.drop_duplicates("stock_id", keep="last"), RealtimeStatus(
            True, quote_date=quote_date, market_open=open_now
        )


def _quote_features(row: pd.Series, market_rate_pp: float) -> dict:
    close = _finite(row.get("close"))
    open_ = _finite(row.get("open"))
    high = _finite(row.get("high"))
    low = _finite(row.get("low"))
    avg = _finite(row.get("average_price"))
    change_pp = _finite(row.get("change_rate"), 0.0)
    vol_ratio = _finite(row.get("volume_ratio"), np.nan)
    amount = _finite(row.get("total_amount"), _finite(row.get("amount"), np.nan))
    buy_p = _finite(row.get("buy_price"))
    sell_p = _finite(row.get("sell_price"))
    buy_v = _finite(row.get("buy_volume"))
    sell_v = _finite(row.get("sell_volume"))

    relative_pp = change_pp - market_rate_pp
    vwap_gap_pp = ((close / avg) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(avg) and avg > 0 else np.nan
    open_move_pp = ((close / open_) - 1.0) * 100.0 if np.isfinite(close) and np.isfinite(open_) and open_ > 0 else np.nan
    day_pos = (close - low) / (high - low) if all(np.isfinite(x) for x in [close, high, low]) and high > low else 0.5
    mid = (buy_p + sell_p) / 2.0 if np.isfinite(buy_p) and np.isfinite(sell_p) and buy_p + sell_p > 0 else np.nan
    spread_pp = ((sell_p - buy_p) / mid) * 100.0 if np.isfinite(mid) and mid > 0 else np.nan
    bid_pressure = buy_v / (buy_v + sell_v) if np.isfinite(buy_v) and np.isfinite(sell_v) and buy_v + sell_v > 0 else 0.5

    # Deliberately bounded, interpretable factor map.  A hot print alone cannot
    # dominate the result; strength must be supported by relative performance,
    # volume and price holding above the intraday average.
    rel_s = _clip01(relative_pp, -2.0, 4.0) * 100
    vwap_s = _clip01(vwap_gap_pp, -1.5, 2.0) * 100
    vol_s = _clip01(vol_ratio, 0.55, 2.5) * 100 if np.isfinite(vol_ratio) else 50.0
    pos_s = _clip01(day_pos, 0.18, 0.88) * 100
    open_s = _clip01(open_move_pp, -2.5, 4.0) * 100
    bid_s = _clip01(bid_pressure, 0.35, 0.65) * 100
    if np.isfinite(amount) and amount > 0:
        # 30m to 2.5b TWD mapped logarithmically, preventing mega-caps from
        # winning solely because of absolute turnover.
        turn_s = _clip01(np.log10(amount), np.log10(30_000_000), np.log10(2_500_000_000)) * 100
    else:
        turn_s = 45.0

    raw = 0.30 * rel_s + 0.20 * vwap_s + 0.18 * vol_s + 0.12 * pos_s + 0.08 * open_s + 0.05 * bid_s + 0.07 * turn_s
    overheat_penalty = 0.0
    if change_pp > 6.0 and np.isfinite(vwap_gap_pp):
        overheat_penalty += min(14.0, max(0.0, (change_pp - 6.0) * 2.0 + max(0.0, vwap_gap_pp - 3.0) * 2.0))
    if day_pos > 0.96 and np.isfinite(vwap_gap_pp) and vwap_gap_pp > 3.0:
        overheat_penalty += 4.0
    spread_penalty = max(0.0, (spread_pp - 0.6) * 6.0) if np.isfinite(spread_pp) else 0.0
    score = float(np.clip(raw - overheat_penalty - spread_penalty, 0, 100))

    if overheat_penalty >= 8.0:
        state = "漲太快，不追"
    elif (relative_pp <= -1.5) or (np.isfinite(vwap_gap_pp) and vwap_gap_pp < -1.2 and day_pos < 0.35):
        state = "轉弱，先略過"
    elif score >= 70 and relative_pp >= 0.8 and (not np.isfinite(vol_ratio) or vol_ratio >= 1.05) and (not np.isfinite(vwap_gap_pp) or vwap_gap_pp >= 0):
        state = "轉強，可優先看"
    elif score >= 60 and (not np.isfinite(vwap_gap_pp) or vwap_gap_pp >= -0.2):
        state = "走勢穩，等拉回"
    elif np.isfinite(vol_ratio) and vol_ratio >= 1.2 and score >= 50:
        state = "有量，還沒突破"
    else:
        state = "等確認"

    reasons: list[str] = []
    if relative_pp >= 1.0:
        reasons.append(f"比大盤強 {relative_pp:.1f} 個百分點")
    elif relative_pp <= -1.0:
        reasons.append(f"比大盤弱 {abs(relative_pp):.1f} 個百分點")
    if np.isfinite(vol_ratio) and vol_ratio >= 1.25:
        reasons.append(f"量能約平常 {vol_ratio:.1f} 倍")
    if np.isfinite(vwap_gap_pp):
        if vwap_gap_pp >= 0.4:
            reasons.append("守在盤中均價上方")
        elif vwap_gap_pp <= -0.5:
            reasons.append("跌到盤中均價下方")
    if not reasons:
        reasons.append("盤中訊號尚未形成明顯優勢")

    return {
        "intraday_score": round(score, 1),
        "state": state,
        "change_rate_pct": round(change_pp, 2),
        "market_change_rate_pct": round(float(market_rate_pp), 2),
        "relative_market_pct_pt": round(float(relative_pp), 2),
        "vwap_gap_pct": None if not np.isfinite(vwap_gap_pp) else round(float(vwap_gap_pp), 2),
        "volume_ratio": None if not np.isfinite(vol_ratio) else round(float(vol_ratio), 2),
        "day_position": round(float(day_pos), 3),
        "spread_pct": None if not np.isfinite(spread_pp) else round(float(spread_pp), 3),
        "bid_pressure": round(float(bid_pressure), 3),
        "price": None if not np.isfinite(close) else float(close),
        "average_price": None if not np.isfinite(avg) else float(avg),
        "open": None if not np.isfinite(open_) else float(open_),
        "high": None if not np.isfinite(high) else float(high),
        "low": None if not np.isfinite(low) else float(low),
        "total_amount": None if not np.isfinite(amount) else float(amount),
        "overheat_penalty": round(float(overheat_penalty), 1),
        "reasons": reasons[:2],
    }


def rerank_snapshot(snap: dict, horizon: str, realtime_df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    if not snap or realtime_df is None or realtime_df.empty:
        return []
    if horizon not in INTRADAY_BLEND:
        horizon = "short"
    rows = {str(r["stock_id"]): r for _, r in realtime_df.iterrows()}
    twse = rows.get("001")
    otc = rows.get("101")
    out: list[dict] = []
    for stock in snap.get("stocks", []):
        ticker = str(stock.get("ticker", ""))
        q = rows.get(_stock_id(ticker))
        if q is None:
            continue
        if market_is_open() and "date" in q:
            qdt = pd.to_datetime(q.get("date"), errors="coerce")
            if pd.notna(qdt) and str(qdt.date()) != _taipei_now().date().isoformat():
                continue
        index_row = otc if ticker.endswith(".TWO") else twse
        market_rate = _finite(index_row.get("change_rate"), 0.0) if index_row is not None else 0.0
        intraday = _quote_features(q, market_rate)
        daily_score = _finite(stock.get("horizons", {}).get(horizon, {}).get("ranking_score"), np.nan)
        if not np.isfinite(daily_score):
            continue
        w = INTRADAY_BLEND[horizon]
        final_score = float(np.clip(daily_score * (1.0 - w) + intraday["intraday_score"] * w, 0, 100))
        item = dict(stock)
        item["intraday"] = intraday
        item["daily_ranking_score"] = round(float(daily_score), 1)
        item["live_ranking_score"] = round(final_score, 1)
        out.append(item)
    out.sort(key=lambda x: x.get("live_ranking_score", -1), reverse=True)
    return out[: max(1, int(top_n))]


def candidate_tickers(snap: dict, per_horizon: int = 40) -> list[str]:
    """Union of the daily model's front-ranked stocks, bounded for realtime calls."""
    if not snap:
        return []
    stocks = [x for x in snap.get("stocks", []) if isinstance(x, dict)]
    chosen: list[str] = []
    for h in ["short", "mid", "long"]:
        ranked = sorted(stocks, key=lambda x: x.get("horizons", {}).get(h, {}).get("ranking_score", -1), reverse=True)
        chosen.extend([str(x.get("ticker")) for x in ranked[:per_horizon] if x.get("ticker")])
    return list(dict.fromkeys(chosen))[:100]
