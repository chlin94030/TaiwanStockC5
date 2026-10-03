"""Alpha Radar V13.4 — Balanced Evidence / mobile-first Taiwan stock research UI."""
from __future__ import annotations

from pathlib import Path
import gc
import html
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from industry_profile import fine_industry
from intraday_engine import FinMindRealtimeClient, candidate_tickers, market_is_open, rerank_snapshot
from market_data import DailyPriceStore, _taipei_timestamp, provider_runtime_status
from policy_engine import HORIZONS
from presentation import investment_view, intraday_view
from return_first_model import holding_review
from trading_calendar import calendar_reference

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))

VIEWS = ["即時", "精選", "短線", "中線", "長線", "個股"]
H_LABEL = {"short": "短線 · 約2週", "mid": "中線 · 約2個月", "long": "長線 · 約6個月"}
REGIME = {"BULL": "大盤偏多", "NEUTRAL": "大盤整理", "BEAR": "大盤偏弱", "UNKNOWN": "大盤待確認"}
FAMILY_LABELS = {
    "完整資料": "full",
    "價格＋公司資料": "business_confirmed",
    "價格＋法人": "flow_confirmed",
    "只看價格": "price_only",
}
HOLD_LABELS = {
    "DATA_UNVERIFIED": "價格尚未核對。",
    "ORIGINAL_STRUCTURE_INVALIDATED": "已跌破原防守價，需優先重新評估。",
    "ORIGINAL_THESIS_INVALIDATED": "原買進理由已不成立，需重新評估。",
    "PROTECTION_TRIGGER_REVIEW_EXECUTION": "已碰到獲利保護價，請依原紀律處理。",
    "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW": "缺少原始防守條件，無法自動判斷。",
    "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "原防守條件尚未被破壞。",
}

CSS = r"""
<style>
:root{
  --paper:#f5f4ef; --surface:#fffefb; --ink:#111318; --muted:#6b6f78;
  --line:#d9d8d2; --blue:#0b57d0; --red:#d92d20; --green:#07875d;
  --amber:#b35c00; --soft:#ecebe5;
}
.stApp{background:var(--paper);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
header[data-testid="stHeader"]{background:rgba(245,244,239,.96);backdrop-filter:blur(8px);}
.block-container{max-width:1120px;padding-top:4.75rem;padding-bottom:4rem;}
@supports (padding-top:env(safe-area-inset-top)){
 .block-container{padding-top:calc(4.75rem + env(safe-area-inset-top));}
}
#MainMenu,footer{visibility:hidden;}
.mast{display:grid;grid-template-columns:1fr auto;align-items:end;border-top:5px solid var(--ink);border-bottom:1px solid var(--ink);padding:13px 1px 11px;margin-bottom:10px;}
.brand{font-size:1.62rem;font-weight:950;letter-spacing:-.055em;line-height:1}.brand span{color:var(--blue)}
.version{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.76rem;color:var(--muted);letter-spacing:.06em;}
.marketbar{display:flex;gap:13px;flex-wrap:wrap;align-items:center;padding:7px 0 10px;border-bottom:1px solid var(--line);font-size:.88rem;color:var(--muted);}
.marketbar b{color:var(--ink)}
.section-kicker{font-size:.72rem;font-weight:900;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin:18px 0 5px;}
.section-title{font-size:1.32rem;font-weight:950;letter-spacing:-.025em;margin:0 0 9px;}
.rankrow{display:grid;grid-template-columns:44px minmax(0,1fr) auto;gap:10px;align-items:center;background:var(--surface);border-top:1px solid var(--ink);border-bottom:1px solid var(--line);padding:13px 10px 12px;margin-top:9px;}
.rankno{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:1.28rem;font-weight:900;color:#a3a39d;align-self:start;padding-top:2px;}
.stockname{font-size:1.28rem;font-weight:950;line-height:1.05;letter-spacing:-.025em}.stockcode{font-size:.79rem;font-weight:750;color:var(--muted);margin-left:5px;}
.industry{font-size:.82rem;color:#4d5159;margin-top:5px;line-height:1.25;}
.quote{text-align:right;min-width:92px}.price{font-size:1.30rem;font-weight:950;line-height:1}.change{font-size:.87rem;font-weight:900;margin-top:5px}.up{color:var(--red)}.down{color:var(--green)}.flat{color:var(--muted)}
.signal{display:grid;grid-template-columns:minmax(130px,.34fr) 1fr;gap:12px;background:var(--surface);padding:9px 10px 11px;border-bottom:1px solid var(--line);}
.signal-title{font-size:1.02rem;font-weight:950;color:var(--blue);line-height:1.3}.signal-copy{font-size:.91rem;line-height:1.45;color:#3f434a}.risk{color:#8a3b11}
.quick{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-top:0;background:var(--surface);}
.q{padding:9px 10px;border-right:1px solid var(--line);min-width:0}.q:last-child{border-right:0}.qk{font-size:.72rem;color:var(--muted);font-weight:800}.qv{font-size:.94rem;font-weight:900;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.liveflag{display:inline-flex;align-items:center;gap:6px;font-size:.78rem;font-weight:900;color:var(--green);letter-spacing:.04em}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 3px rgba(7,135,93,.12)}
.offflag{font-size:.78rem;font-weight:850;color:var(--muted)}
.datagrid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--surface);margin:8px 0 12px}.datum{padding:10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.datum:nth-child(4n){border-right:0}.dk{font-size:.74rem;color:var(--muted);font-weight:800}.dv{font-size:1rem;font-weight:950;margin-top:3px}.ds{font-size:.73rem;color:var(--muted);margin-top:2px;line-height:1.3}
.plan{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin-bottom:8px}.plan>div{background:var(--surface);padding:9px}.plan small{display:block;color:var(--muted);font-weight:800}.plan b{display:block;margin-top:3px;font-size:.98rem}
.chartchips{display:flex;gap:12px;align-items:center;font-size:.76rem;color:var(--muted);margin:4px 0 1px}.chipline{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:5px}.ma20{background:#0b57d0}.ma60{background:#f08c00}
.mini-note{font-size:.78rem;color:var(--muted);line-height:1.4}
[data-testid="stExpander"]{border:0!important;border-bottom:1px solid var(--line)!important;border-radius:0!important;background:transparent!important;}[data-testid="stExpander"] summary{padding-left:0!important;padding-right:0!important;font-weight:850!important;}
div[data-baseweb="tab-list"]{gap:4px}button[data-baseweb="tab"]{font-weight:850}
.stButton>button{border-radius:5px!important;font-weight:850!important;min-height:42px}.stButton>button[kind="primary"]{background:var(--ink)!important;border-color:var(--ink)!important;}
hr{border-color:var(--line)!important}
@media(max-width:720px){
 .stApp{font-size:17.5px}.block-container{padding:4.35rem .72rem 3rem}.mast{padding:13px 0 10px;margin-top:.15rem}.brand{font-size:1.52rem}.version{font-size:.68rem}
 .marketbar{font-size:.84rem;gap:8px 11px}.rankrow{grid-template-columns:36px minmax(0,1fr) auto;padding:12px 7px 11px;gap:7px}.rankno{font-size:1.08rem}.stockname{font-size:1.30rem}.stockcode{font-size:.78rem}.industry{font-size:.84rem}.quote{min-width:78px}.price{font-size:1.25rem}.change{font-size:.88rem}
 .signal{grid-template-columns:1fr;gap:3px;padding:9px 7px 10px}.signal-title{font-size:1.05rem}.signal-copy{font-size:.96rem;line-height:1.42}
 .quick{grid-template-columns:repeat(2,1fr)}.q:nth-child(2){border-right:0}.q:nth-child(-n+2){border-bottom:1px solid var(--line)}.qk{font-size:.78rem}.qv{font-size:1rem}
 .datagrid{grid-template-columns:repeat(2,1fr)}.datum:nth-child(odd){border-right:1px solid var(--line)}.datum:nth-child(even){border-right:0}.dk{font-size:.80rem}.dv{font-size:1.05rem}.ds{font-size:.77rem}
 .plan{grid-template-columns:repeat(2,1fr)}.section-title{font-size:1.25rem}
 .stRadio [role="radiogroup"]{gap:.2rem!important}.stRadio label{font-size:.92rem!important}
 @supports (padding-top:env(safe-area-inset-top)){.block-container{padding-top:calc(4.35rem + env(safe-area-inset-top));}}
}
</style>
"""


def esc(v) -> str:
    return html.escape(str(v if v is not None else ""))


def finite(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def nfmt(v, d=2):
    x = finite(v)
    if x is None:
        return "—"
    return f"{x:,.{d}f}".rstrip("0").rstrip(".")


def pp(v, d=1):
    x = finite(v)
    return "—" if x is None else f"{x:+.{d}f}%"


def decpct(v, d=1):
    x = finite(v)
    return "—" if x is None else f"{x*100:+.{d}f}%"


def lots(v):
    x = finite(v)
    return "—" if x is None else f"{x:+,.0f}張"


def _research(stock: dict):
    r = stock.get("research") or {}
    return (
        r.get("monthly_revenue") or {},
        r.get("financials") or {},
        r.get("valuation") or {},
        r.get("institutional_flow") or {},
        r.get("main_force_proxy") or {},
    )


def _quick_values(stock: dict):
    rev, fin, val, inst, _ = _research(stock)
    latest = rev.get("latest") or {}
    return [
        ("月營收", f"{nfmt(latest.get('revenue_billion'),1)}億", f"YoY {pp(latest.get('yoy_pct'))}"),
        ("近4季EPS", nfmt(fin.get("eps_4q_sum"), 2), "單季加總"),
        ("本益比", nfmt(val.get("pe"), 1), "倍"),
        ("近月法人", lots(inst.get("total_net_lots")) if inst.get("available") else "—", "合計"),
    ]


def _quick_html(stock: dict) -> str:
    cells = []
    for k, v, s in _quick_values(stock):
        cells.append(f"<div class='q'><div class='qk'>{esc(k)}</div><div class='qv'>{esc(v)}</div><div class='ds'>{esc(s)}</div></div>")
    return "<div class='quick'>" + "".join(cells) + "</div>"


def _render_research(stock: dict):
    rev, fin, val, inst, main = _research(stock)
    latest = rev.get("latest") or {}
    qs = fin.get("quarters") or []
    gm = fin.get("gross_margin_latest_pct")
    items = [
        ("最新月營收", f"{nfmt(latest.get('revenue_billion'))} 億", latest.get("month") or "—"),
        ("月增", pp(latest.get("mom_pct")), "MoM"),
        ("年增", pp(latest.get("yoy_pct")), "YoY"),
        ("近4季EPS", nfmt(fin.get("eps_4q_sum")), "單季加總"),
        ("最新毛利率", pp(gm, 1) if gm is not None else "—", ""),
        ("本益比", f"{nfmt(val.get('pe'),1)} 倍" if val.get("pe") is not None else "—", val.get("date") or ""),
        ("外資", lots(inst.get("foreign_net_lots")) if inst.get("available") else "—", "近月"),
        ("投信", lots(inst.get("trust_net_lots")) if inst.get("available") else "—", "近月"),
        ("自營商", lots(inst.get("dealer_net_lots")) if inst.get("available") else "—", "近月"),
        ("三大法人", lots(inst.get("total_net_lots")) if inst.get("available") else "—", "近月合計"),
        ("主力代理", lots(main.get("proxy_net_lots")) if main.get("available") else "未啟用", "券商分點代理"),
        ("近4季平均毛利", pp(fin.get("gross_margin_4q_avg_pct"),1) if fin.get("gross_margin_4q_avg_pct") is not None else "—", ""),
    ]
    html_cells = "".join(
        f"<div class='datum'><div class='dk'>{esc(k)}</div><div class='dv'>{esc(v)}</div><div class='ds'>{esc(s)}</div></div>"
        for k, v, s in items
    )
    st.markdown(f"<div class='datagrid'>{html_cells}</div>", unsafe_allow_html=True)

    if qs:
        rows = []
        for q in qs[-4:]:
            rows.append({
                "季度": q.get("quarter"),
                "EPS": q.get("eps"),
                "同年度累積EPS*": q.get("eps_ytd_sum_unadjusted"),
                "毛利率(%)": q.get("gross_margin_pct"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("* 依單季 EPS 加總；遇配股/分割可能與公司重編累計值不同。")
    if inst.get("summary"):
        st.caption(str(inst.get("summary")))
    if main.get("summary"):
        st.caption(str(main.get("summary")))


def _chart_dataframe(chart: dict) -> pd.DataFrame:
    if not chart or "ohlcv" not in chart:
        return pd.DataFrame()
    df = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(chart.get("dates", []), errors="coerce")
    df = df.dropna(subset=["Date"]).reset_index(drop=True)
    if df.empty:
        return df
    c = pd.to_numeric(df["Close"], errors="coerce")
    df["MA20"] = c.rolling(20).mean()
    df["MA60"] = c.rolling(60).mean()
    return df


def render_chart(chart: dict, plan: dict | None, key: str):
    raw = _chart_dataframe(chart)
    if raw.empty:
        st.caption("K線資料未取得。")
        return
    window = st.radio("區間", ["1個月", "3個月", "6個月", "1年"], index=1, horizontal=True, label_visibility="collapsed", key=f"{key}_range")
    n = {"1個月": 24, "3個月": 66, "6個月": 132, "1年": 252}[window]
    df = raw.tail(n).copy()
    x = df["Date"].dt.strftime("%Y-%m-%d").tolist()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.79, 0.21], vertical_spacing=0.025)
    fig.add_trace(go.Candlestick(
        x=x, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        increasing_line_color="#d92d20", increasing_fillcolor="#d92d20",
        decreasing_line_color="#07875d", decreasing_fillcolor="#07875d",
        name="價格", showlegend=False, whiskerwidth=.35,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df.MA20, mode="lines", line=dict(color="#0b57d0", width=2.1), showlegend=False, hovertemplate="20日 %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df.MA60, mode="lines", line=dict(color="#f08c00", width=2.1), showlegend=False, hovertemplate="60日 %{y:.2f}<extra></extra>"), row=1, col=1)
    colors = np.where(df.Close >= df.Open, "rgba(217,45,32,.62)", "rgba(7,135,93,.62)")
    fig.add_trace(go.Bar(x=x, y=df.Volume, marker_color=colors, showlegend=False, hovertemplate="量 %{y:,.0f}<extra></extra>"), row=2, col=1)

    if plan:
        stop = finite(plan.get("invalidation")); trigger = finite(plan.get("trigger"))
        if trigger is not None:
            fig.add_hline(y=trigger, line_color="#8a5a00", line_width=1.2, line_dash="dot", row=1, col=1)
        if stop is not None:
            fig.add_hline(y=stop, line_color="#9b1c1c", line_width=1.2, line_dash="dash", row=1, col=1)

    vals = pd.concat([df["High"], df["Low"], df["MA20"], df["MA60"]]).dropna()
    if not vals.empty:
        lo, hi = float(vals.min()), float(vals.max())
        pad = max((hi - lo) * .20, max(hi, 1) * .012)
        fig.update_yaxes(range=[max(0, lo - pad), hi + pad], row=1, col=1)

    # Three labels on mobile is enough; hover carries exact date.
    if x:
        ids = sorted(set(np.linspace(0, len(x)-1, min(3, len(x)), dtype=int).tolist()))
        tickvals = [x[i] for i in ids]
        ticktext = [pd.to_datetime(x[i]).strftime("%m/%d") for i in ids]
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, row=2, col=1)

    fig.update_layout(
        height=340, margin=dict(l=3, r=5, t=5, b=3), paper_bgcolor="#fffefb", plot_bgcolor="#fffefb",
        xaxis_rangeslider_visible=False, dragmode=False, hovermode="x unified", showlegend=False,
        font=dict(size=11, color="#4d5159"),
    )
    fig.update_xaxes(showgrid=False, fixedrange=True)
    fig.update_yaxes(gridcolor="#ecebe5", zeroline=False, fixedrange=True, tickfont=dict(size=10))
    st.markdown("<div class='chartchips'><span><i class='chipline ma20'></i>20日</span><span><i class='chipline ma60'></i>60日</span></div>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False, "scrollZoom": False, "displaylogo": False})


def _history_line(stock: dict, h: str) -> str:
    f = stock.get("horizons", {}).get(h, {}).get("forecast", {}) or {}
    s = f.get("strategy") or {}
    if not f.get("estimate_available"):
        return "歷史案例不足"
    med = decpct(s.get("median"), 1)
    rate = finite(s.get("smoothed_positive_rate"))
    n = int(finite(f.get("local_effective_n")) or 0)
    return f"相似案例 {n} 次 · 中位數 {med} · 偏正向 {rate*100:.0f}%" if rate is not None else f"相似案例 {n} 次 · 中位數 {med}"


def _row(stock: dict, h: str, snap: dict, rank: int, live: bool = False):
    name = stock.get("name") or stock.get("ticker")
    ticker = stock.get("ticker") or ""
    fine = stock.get("fine_industry") or fine_industry(ticker, name, stock.get("industry", ""))
    role_label = ((stock.get("role") or {}).get("label") or "").strip()
    stage_label = ((stock.get("stage") or {}).get("label") or "").strip()
    context_line = " · ".join([x for x in [fine, role_label, stage_label] if x])
    if live:
        live_info = stock.get("intraday") or {}
        view = intraday_view(stock)
        price = live_info.get("price") or stock.get("price")
        change = finite(live_info.get("change_rate_pct"))
        reasons = view.get("reasons") or []
        title = view.get("title") or "即時資料不足"
        risk = ""
        history = _history_line(stock, h)
    else:
        view = investment_view(stock, h, snap.get("market") or {})
        price = stock.get("price")
        change = None
        reasons = view.get("reasons") or []
        title = view.get("title") or "觀察"
        risk = view.get("risk") or ""
        history = _history_line(stock, h)

    cls = "flat"
    if change is not None:
        cls = "up" if change > 0 else "down" if change < 0 else "flat"
    change_text = pp(change, 2) if change is not None else esc(H_LABEL[h])
    reason_text = " · ".join(str(x) for x in reasons[:2])
    risk_text = f" · <span class='risk'>{esc(risk)}</span>" if risk else ""
    st.markdown(f"""
    <div class="rankrow">
      <div class="rankno">{rank:02d}</div>
      <div><div><span class="stockname">{esc(name)}</span><span class="stockcode">{esc(ticker)}</span></div><div class="industry">{esc(context_line)}</div></div>
      <div class="quote"><div class="price">{nfmt(price)}</div><div class="change {cls}">{esc(change_text)}</div></div>
    </div>
    <div class="signal"><div class="signal-title">{esc(title)}</div><div class="signal-copy">{esc(reason_text)}{risk_text}<div class="mini-note">{esc(history)}</div></div></div>
    {_quick_html(stock)}
    """, unsafe_allow_html=True)

    label = f"{name}｜數據與走勢"
    with st.expander(label, expanded=False):
        _render_research(stock)
        block = stock.get("horizons", {}).get(h, {}) or {}
        plan = block.get("plan") or {}
        if plan:
            pos = block.get("position_guidance") or {}
            pos_label = pos.get("label") or "依風險調整"
            st.markdown(f"""<div class='plan'>
              <div><small>較理想區間</small><b>{nfmt(plan.get('zone_low'))}–{nfmt(plan.get('zone_high'))}</b></div>
              <div><small>轉強確認</small><b>{nfmt(plan.get('trigger'))}</b></div>
              <div><small>不追高超過</small><b>{nfmt(plan.get('chase_limit'))}</b></div>
              <div><small>跌破重看</small><b>{nfmt(plan.get('invalidation'))}</b></div>
              <div><small>目前部位節奏</small><b>{esc(pos_label)}</b></div>
            </div>""", unsafe_allow_html=True)
        chart = None
        try:
            chart = service.chart_on_demand(snap, ticker, DATA_DIR, allow_fetch=False)
        except Exception:
            pass
        render_chart(chart, plan, f"chart_{'live' if live else h}_{ticker}_{snap.get('snapshot_id','x')}")


def _render_daily_list(snap: dict | None, h: str, n: int = 5):
    if not snap:
        st.info("請先更新市場資料。")
        return
    picks = service.select_market_best(snap, h, n=n)
    if not picks:
        st.info("目前沒有可排序標的。")
        return
    for i, stock in enumerate(picks, 1):
        _row(stock, h, snap, i, live=False)


def _market_header(snap: dict | None):
    if not snap:
        st.markdown("<div class='marketbar'><span>尚未建立市場快照</span></div>", unsafe_allow_html=True)
        return
    m = snap.get("market") or {}
    cov = snap.get("coverage") or {}
    off = cov.get("official_eod") or {}
    parts = [
        f"<span><b>{esc(snap.get('price_date') or '—')}</b> 盤後</span>",
        f"<span>{esc(REGIME.get(m.get('regime'), '大盤待確認'))}</span>",
        f"<span>上市 {esc(off.get('twse_date') or '—')}</span>",
        f"<span>上櫃 {esc(off.get('tpex_date') or '—')}</span>",
    ]
    stale = int(cov.get("stale_excluded") or 0)
    if stale:
        parts.append(f"<span>{stale:,} 檔舊資料已排除</span>")
    st.markdown("<div class='marketbar'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


def _intraday_render_once(snap: dict):
    now = _taipei_timestamp()
    if not market_is_open(now):
        t = now.strftime("%H:%M")
        label = "盤前" if now.hour < 9 else "收盤後"
        st.markdown(f"<div class='offflag'>{label} · {esc(t)} · 09:00–13:30 才啟用盤中排序</div>", unsafe_allow_html=True)
        _render_daily_list(snap, "short", n=8)
        return

    client = FinMindRealtimeClient()
    if not client.configured:
        st.warning("盤中即時資料需要 FinMind Sponsor Token。盤後模型仍可正常使用。")
        _render_daily_list(snap, "short", n=8)
        return

    tickers = candidate_tickers(snap, per_horizon=40)
    df, status = client.snapshots(tickers)
    if not status.available:
        st.warning(f"即時資料暫不可用：{status.reason}")
        _render_daily_list(snap, "short", n=8)
        return
    live = rerank_snapshot(snap, "short", df, top_n=8)
    twii = df[df["stock_id"].astype(str) == "001"] if "stock_id" in df.columns else pd.DataFrame()
    idx_change = finite(twii.iloc[-1].get("change_rate")) if not twii.empty else None
    st.markdown(
        f"<div class='marketbar'><span class='liveflag'><i class='dot'></i>盤中更新</span><span>{esc(now.strftime('%H:%M'))}</span><span>加權 {esc(pp(idx_change,2))}</span><span>候選 {len(tickers)} 檔</span></div>",
        unsafe_allow_html=True,
    )
    if not live:
        st.info("即時資料已取得，但候選股目前無可排序資料。")
        return
    for i, stock in enumerate(live, 1):
        _row(stock, "short", snap, i, live=True)


# Only this fragment refreshes; the expensive multi-year model remains untouched.
if hasattr(st, "fragment"):
    @st.fragment(run_every=60)
    def intraday_fragment(snap: dict):
        _intraday_render_once(snap)
else:
    def intraday_fragment(snap: dict):
        _intraday_render_once(snap)


def _prime(snap: dict | None):
    if not snap:
        st.info("請先更新市場資料。")
        return
    used: set[str] = set()
    for h in HORIZONS:
        picks = service.select_market_best(snap, h, n=8)
        stock = next((x for x in picks if x.get("ticker") not in used), picks[0] if picks else None)
        if stock:
            used.add(str(stock.get("ticker")))
            st.markdown(f"<div class='section-kicker'>{esc(H_LABEL[h])}</div>", unsafe_allow_html=True)
            _row(stock, h, snap, 1, live=False)


def _doctor(snap: dict | None):
    with st.form("doctor_v133"):
        code = st.text_input("股票代碼", value="2330")
        h = st.selectbox("時間", list(H_LABEL), format_func=lambda x: H_LABEL[x], index=1)
        own = st.checkbox("已有持股")
        c1, c2 = st.columns(2)
        with c1:
            cost = st.number_input("成本", min_value=0.0, value=0.0)
            stop = st.number_input("原防守價", min_value=0.0, value=0.0)
        with c2:
            trail = st.number_input("獲利保護價", min_value=0.0, value=0.0)
            thesis = st.selectbox("原買進理由", ["尚未確認", "仍成立", "已不成立"])
        go_ = st.form_submit_button("查看", type="primary", use_container_width=True)
    if go_:
        dr = service.diagnose(code, snap, DATA_DIR)
        dr["h"] = h
        st.session_state["alpha_doctor"] = dr
    dr = st.session_state.get("alpha_doctor")
    if not dr:
        return
    if dr.get("error"):
        st.error("找不到足夠行情資料。")
        return
    stock = dr.get("stock")
    if not stock:
        return
    _row(stock, dr.get("h", "mid"), snap or {"market": {}, "snapshot_id": "doctor"}, 1, live=False)
    if own:
        result = holding_review(
            stock.get("price", 0),
            original_invalidation=stop if stop > 0 else None,
            trailing_protection=trail if trail > 0 else None,
            thesis_broken=(thesis == "已不成立"),
        )
        pnl = (float(stock.get("price")) / cost - 1) * 100 if cost > 0 and finite(stock.get("price")) is not None else None
        st.info(HOLD_LABELS.get(result, result) + (f" 目前損益約 {pnl:+.1f}%" if pnl is not None else ""))


def main():
    st.set_page_config(page_title="Alpha Radar 13.4", page_icon="◼", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("<div class='mast'><div class='brand'>ALPHA<span>/TW</span></div><div class='version'>BALANCED 13.4</div></div>", unsafe_allow_html=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if st.session_state.get("alpha_version") != service.OPERATIONS_VERSION:
        for k in ["alpha_snapshot", "alpha_doctor", "alpha_error"]:
            st.session_state.pop(k, None)
        st.session_state["alpha_version"] = service.OPERATIONS_VERSION
        previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json")
        if previous and previous.get("model_version") == service.OPERATIONS_VERSION:
            st.session_state["alpha_snapshot"] = service.compact_session_dashboard(previous)

    with st.sidebar:
        st.markdown("### 設定")
        family_label = st.selectbox("資料範圍", list(FAMILY_LABELS), index=0)
        period = st.selectbox("歷史長度", ["5y", "8y", "10y", "3y"], index=0)
        candidate_size = st.selectbox("比較股票數", [400, 700, 1000], index=2)
        research_pool = st.selectbox("公司資料檔數", [5, 10, 15], index=1)
        status = provider_runtime_status()
        st.caption("FinMind Token：" + ("已設定" if status.get("finmind_token_configured") else "未設定"))
        st.caption("盤中：需要 Sponsor 即時權限")
        if st.button("清除快取"):
            DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
            service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
            st.session_state.pop("alpha_snapshot", None)
            st.success("已清除")

    settings = service.RunSettings(
        reference_size=600,
        candidate_size=int(candidate_size),
        research_pool_per_horizon=int(research_pool),
        history_period=period,
        model_family=FAMILY_LABELS[family_label],
    )

    if st.button("更新盤後模型", type="primary", use_container_width=True):
        started = time.monotonic()
        bar = st.progress(0, text="0% · 準備更新")
        try:
            def update(stage, value):
                v = min(1.0, max(0.0, float(value)))
                elapsed = int(time.monotonic() - started)
                bar.progress(v, text=f"{int(v*100):d}% · {stage} · {elapsed}s")
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            st.session_state["alpha_snapshot"] = service.compact_session_dashboard(snap)
            st.session_state.pop("alpha_error", None)
            gc.collect()
        except Exception as exc:
            st.session_state["alpha_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            bar.empty()

    snap = st.session_state.get("alpha_snapshot")
    _market_header(snap)
    if st.session_state.get("alpha_error"):
        st.error(st.session_state.get("alpha_error"))

    view = st.radio("功能", VIEWS, horizontal=True, label_visibility="collapsed", key="view_v133")
    if view == "即時":
        st.markdown("<div class='section-kicker'>LIVE PULSE</div><div class='section-title'>盤中雷達</div>", unsafe_allow_html=True)
        if not snap:
            st.info("先建立盤後模型，盤中才有候選池。")
        else:
            intraday_fragment(snap)
    elif view == "精選":
        st.markdown("<div class='section-kicker'>PRIME</div><div class='section-title'>三個時間尺度</div>", unsafe_allow_html=True)
        _prime(snap)
    elif view in {"短線", "中線", "長線"}:
        h = {"短線": "short", "中線": "mid", "長線": "long"}[view]
        st.markdown(f"<div class='section-kicker'>{esc(H_LABEL[h])}</div><div class='section-title'>前 5 名</div>", unsafe_allow_html=True)
        _render_daily_list(snap, h, 5)
    else:
        st.markdown("<div class='section-kicker'>CHECK</div><div class='section-title'>個股診斷</div>", unsafe_allow_html=True)
        _doctor(snap)

    st.caption("歷史案例用於比較，不代表未來結果；盤中訊號只重排已通過盤後模型的候選股。")


if __name__ == "__main__":
    main()
