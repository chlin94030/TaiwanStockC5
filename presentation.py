"""Plain-language decision copy for Alpha Radar V13.3.

Keep the visible layer terse.  Technical names stay in the model/debug layer,
not in the recommendation headline.
"""
from __future__ import annotations


def _f(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


STATE_VIEW = {
    "CONDITIONS_MET_NOT_FILLED": ("可看", "價格已到可觀察區"),
    "WAIT_ENTRY_ZONE": ("等拉回", "條件不差，價格再低一些更好"),
    "WAIT_BREAKOUT": ("等轉強", "還沒突破，先等"),
    "WAIT_CONFIRMATION": ("再確認", "價格與量還沒一起轉強"),
    "DO_NOT_CHASE": ("不追", "走勢強，但現在太遠"),
    "INVALIDATED": ("先略過", "近期結構已轉弱"),
    "DATA_UNVERIFIED": ("資料待更新", "先取得最新行情"),
    "NO_RETURN_ESTIMATE": ("先觀察", "可比歷史案例不足"),
}


def _append_unique(dst: list[str], text: str):
    if text and text not in dst:
        dst.append(text)


def investment_view(stock: dict, horizon: str, market: dict | None = None) -> dict:
    market = market or {}
    block = (stock.get("horizons") or {}).get(horizon, {}) or {}
    f = block.get("forecast") or {}
    s = f.get("strategy") or {}
    research = stock.get("research") or {}
    rev = research.get("monthly_revenue") or {}
    fin = research.get("financials") or {}
    inst = research.get("institutional_flow") or {}

    title, action = STATE_VIEW.get(str(block.get("entry_state") or ""), ("觀察", "目前沒有明顯優勢"))
    score = _f(block.get("ranking_score"), 0.0) or 0.0
    median = _f(s.get("median"))
    positive = _f(s.get("smoothed_positive_rate"), _f(s.get("historical_positive_rate")))
    alpha = _f(f.get("alpha_mean"))

    good: list[str] = []
    risk: list[str] = []

    setup = str(stock.get("setup") or "")
    if setup == "WASHOUT":
        _append_unique(good, "拉回後已重新站穩")
    elif setup == "RECLAIM":
        _append_unique(good, "跌破後已重新站回")
    elif setup == "BREAKOUT":
        _append_unique(good, "價格正嘗試突破近期高點")

    # 1) price/market evidence
    if alpha is not None:
        if alpha >= 0.05:
            _append_unique(good, f"近20日比大盤強 {alpha*100:.1f} 個百分點")
        elif alpha >= 0.015:
            _append_unique(good, "近期走勢比大盤強")
        elif alpha <= -0.04:
            _append_unique(risk, "近期明顯弱於大盤")

    # 2) historical analog evidence; use smoothed rate to avoid tiny-sample drama
    if median is not None and positive is not None:
        if median >= 0.025 and positive >= 0.58:
            _append_unique(good, f"過去相似走勢多數偏正向（約 {positive*100:.0f}%）")
        elif median < -0.015 or positive < 0.46:
            _append_unique(risk, "過去相似走勢優勢不明顯")

    # 3) revenue
    latest = rev.get("latest") or {}
    yoy = _f(latest.get("yoy_pct"))
    if yoy is not None:
        if yoy >= 12:
            _append_unique(good, f"最新月營收年增 {yoy:.1f}%")
        elif yoy <= -10:
            _append_unique(risk, f"最新月營收年減 {abs(yoy):.1f}%")

    # 4) EPS / margin
    qs = fin.get("quarters") or []
    eps = [_f(q.get("eps")) for q in qs if _f(q.get("eps")) is not None]
    if len(eps) >= 2:
        if eps[-1] > 0 and eps[-1] >= eps[-2] * 1.12:
            _append_unique(good, "最新一季獲利較前季改善")
        elif eps[-1] < 0:
            _append_unique(risk, "最新一季仍虧損")
        elif eps[-1] <= eps[-2] * 0.75:
            _append_unique(risk, "最新一季獲利明顯降溫")

    gm = _f(fin.get("gross_margin_latest_pct"))
    gm4 = _f(fin.get("gross_margin_4q_avg_pct"))
    if gm is not None and gm4 is not None:
        if gm >= gm4 + 1.5:
            _append_unique(good, "最新毛利率高於近四季平均")
        elif gm <= gm4 - 2.0:
            _append_unique(risk, "最新毛利率低於近四季平均")

    # 5) institutional flow
    net = _f(inst.get("total_net_lots")) if inst.get("available") else None
    if net is not None:
        if net >= 3000:
            _append_unique(good, "近一月三大法人偏買超")
        elif net <= -3000:
            _append_unique(risk, "近一月三大法人偏賣超")

    if market.get("regime") == "BEAR" and horizon == "short":
        _append_unique(risk, "大盤短線偏弱")

    if score < 48:
        title = "先略過" if title not in {"不追", "資料待更新"} else title
        _append_unique(risk, "整體條件目前不突出")
    elif score >= 72 and title in {"可看", "等轉強", "再確認"}:
        title = "優先看" if title == "可看" else title

    if not good:
        good = ["目前沒有足夠明顯的加分項"]

    return {
        "title": title,
        "action": action,
        "reasons": good[:2],
        "risk": risk[0] if risk else "",
    }


def intraday_view(stock: dict) -> dict:
    live = stock.get("intraday") or {}
    return {
        "title": live.get("state") or "即時資料不足",
        "reasons": (live.get("reasons") or ["即時資料不足"])[:2],
    }


def plain_summary(forecast: dict) -> str:
    if not forecast or not forecast.get("estimate_available"):
        return "可比歷史案例不足"
    s = forecast.get("strategy") or {}
    median = _f(s.get("median"), 0.0) or 0.0
    rate = _f(s.get("smoothed_positive_rate"), _f(s.get("historical_positive_rate"), 0.0)) or 0.0
    if median > 0 and rate >= 0.55:
        return f"相似走勢約 {rate*100:.0f}% 偏正向"
    return "相似走勢沒有明顯優勢"
