"""Offline regression + stress tests for Alpha Radar V13.4. No network required."""
from __future__ import annotations

import datetime
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    import yfinance  # noqa
except Exception:
    yf_stub = types.ModuleType("yfinance")
    def _offline(*args, **kwargs):
        raise RuntimeError("live download disabled in offline self-test")
    yf_stub.download = _offline
    yf_stub.Ticker = _offline
    sys.modules["yfinance"] = yf_stub

import market_data as market_data_module
from market_data import ResearchDataClient, DailyPriceStore, _yf_history_args
from policy_engine import generate_trade_plan, evaluate_entry_state
from return_first_model import estimate_horizon_return
from radar_service import (
    _fundamental_score, _flow_score, _combined_ranking_score, _decorate_role_stage,
    _position_guidance, _liquidity_score, select_market_best,
)
from industry_profile import fine_industry
from intraday_engine import _quote_features, rerank_snapshot, candidate_tickers, market_is_open
from presentation import investment_view, intraday_view
from trading_calendar import calendar_reference
from validate_100 import summarize as validation_summarize


def synthetic_prices(seed=42, n=1250, drift=0.00055, vol=0.017):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    ret = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(ret))
    op = close * (1 + rng.normal(0, 0.0035, n))
    high = np.maximum(op, close) * (1 + rng.uniform(0.001, 0.014, n))
    low = np.minimum(op, close) * (1 - rng.uniform(0.001, 0.014, n))
    volume = rng.lognormal(np.log(2_000_000), 0.35, n)
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


class FakeClient(ResearchDataClient):
    def _finmind(self, dataset, stock_id, start_date, end_date=None, ttl_hours=6.0):
        if dataset == "TaiwanStockMonthRevenue":
            rows=[]; base=pd.Period("2024-09",freq="M")
            for i in range(25):
                p=base+i
                rows.append({"date":p.start_time.strftime("%Y-%m-%d"),"stock_id":stock_id,"revenue":10e9*(1+i*.02),"revenue_month":p.month,"revenue_year":p.year})
            return rows
        if dataset == "TaiwanStockFinancialStatements":
            rows=[]
            for d,eps,rev,gp in [("2025-12-31",2,100e9,40e9),("2026-03-31",2.4,105e9,43e9),("2026-06-30",2.8,112e9,47e9),("2026-09-30",3.1,120e9,51e9)]:
                for typ,val in [("EPS",eps),("Revenue",rev),("GrossProfit",gp)]: rows.append({"date":d,"stock_id":stock_id,"type":typ,"value":val})
            return rows
        if dataset == "TaiwanStockPER":
            return [{"date":"2026-09-30","stock_id":stock_id,"PER":"24.5","PBR":"4.2","dividend_yield":"2.1"}]
        if dataset == "TaiwanStockInstitutionalInvestorsBuySellWide":
            return [{"date":f"2026-09-{i+1:02d}","stock_id":stock_id,"Foreign_Investor_buy":2e6,"Foreign_Investor_sell":1.5e6,"Foreign_Dealer_Self_buy":0,"Foreign_Dealer_Self_sell":0,"Investment_Trust_buy":.5e6,"Investment_Trust_sell":.35e6,"Dealer_buy":.3e6,"Dealer_sell":.25e6} for i in range(20)]
        return []


def test_model_core():
    assert "start" in _yf_history_args("8y")
    assert _yf_history_args("5y") == {"period":"5y"}
    bench = synthetic_prices(900, drift=0.00025, vol=0.011)
    stock = synthetic_prices(42, drift=0.00075, vol=0.018)
    settings = SimpleNamespace(commission=.001425,sell_tax=.003,slippage=.0005)
    for h in ["short","mid","long"]:
        f = estimate_horizon_return(stock,h,settings,twii_ret_20d=.01,benchmark_df=bench)
        p = generate_trade_plan(stock,h)
        assert f["estimate_available"] is True
        assert 0 <= f["technical_factor_score"] <= 100
        assert 0 <= f["empirical_quality_score"] <= 100
        assert 0 < f["local_effective_n"] <= 100
        assert "smoothed_positive_rate" in f["strategy"]
        assert 0 <= f["features"]["kd_k"] <= 100
        assert 0 <= f["features"]["kd_d"] <= 100
        assert "macd_pct" in f["features"]
        assert 0 <= p["entry_score"] <= 98
        assert p["invalidation"] < float(stock["Close"].iloc[-1])
        assert p["invalidation"] < float(stock.Close.iloc[-1])
        assert p["zone_low"] <= p["zone_high"]
        assert evaluate_entry_state(stock,p) in {"CONDITIONS_MET_NOT_FILLED","WAIT_ENTRY_ZONE","WAIT_BREAKOUT","DO_NOT_CHASE","INVALIDATED"}

    # Benchmark alignment regression: changing only historical benchmark should
    # change analog selection/relative-strength evidence, proving V13.2's static
    # current-market subtraction is gone.
    bear_bench = bench.copy()
    bear_bench["Close"] = 250 * np.exp(np.cumsum(np.random.default_rng(91).normal(-.00035,.012,len(bear_bench))))
    a = estimate_horizon_return(stock,"mid",settings,benchmark_df=bench)
    b = estimate_horizon_return(stock,"mid",settings,benchmark_df=bear_bench)
    assert a.get("analog_median_distance") != b.get("analog_median_distance") or a.get("empirical_quality_score") != b.get("empirical_quality_score")


def test_100_series_stress():
    # Debug stress only — intentionally NOT labelled real-stock validation.
    bench = synthetic_prices(777, n=900, drift=.00025, vol=.012)
    settings = SimpleNamespace(commission=.001425,sell_tax=.003,slippage=.0005)
    ok=0
    for seed in range(100):
        df=synthetic_prices(1000+seed,n=900,drift=.00015+(seed%11)*.00007,vol=.012+(seed%7)*.001)
        f=estimate_horizon_return(df,"short",settings,benchmark_df=bench)
        if f.get("estimate_available") and 0 <= f.get("technical_factor_score",-1) <= 100:
            ok += 1
    assert ok == 100


def test_research_and_copy():
    with tempfile.TemporaryDirectory() as td:
        r=FakeClient(Path(td)/"research.sqlite").stock_research("2330.TW")
        assert len(r["monthly_revenue"]["rows"])==12
        assert abs(r["financials"]["eps_4q_sum"]-10.3)<1e-9
        assert r["valuation"]["pe"]==24.5
        fs,_=_fundamental_score(r); fl,_=_flow_score(r,2_000_000)
        assert fs is not None and fl is not None
        stock={"ticker":"2330.TW","name":"台積電","industry":"半導體業","fine_industry":fine_industry("2330.TW","台積電","半導體業"),"avg_volume_20d":2e6,"research":r,"horizons":{}}
        for h in ["short","mid","long"]:
            stock["horizons"][h]={"entry_state":"CONDITIONS_MET_NOT_FILLED","forecast":{"estimate_available":True,"technical_factor_score":70,"composite_factor_score":70,"empirical_quality_score":68,"alpha_mean":.03,"local_effective_n":55,"strategy":{"median":.04,"mean":.05,"p10":-.04,"expected_shortfall10_loss":-.07,"historical_positive_rate":.62,"smoothed_positive_rate":.60}}}
            stock["horizons"][h]["plan"]={"ma5":105,"ma20":100,"ma60":95,"ma120":90}
            _combined_ranking_score(stock,h,"full")
            v=investment_view(stock,h,{"regime":"BULL"})
            assert v["title"] and v["action"] and 1 <= len(v["reasons"]) <= 2
        stock.update({"price":108,"setup":"TREND"})
        _decorate_role_stage(stock)
        assert stock["role"]["label"] in {"核心型","成長型","短打型"}
        assert stock["stage"]["label"] in {"整理","基本面轉折","突破","趨勢成形","偏離過大","過熱"}
    assert fine_industry("2330.TW","台積電","半導體業") == "半導體－晶圓代工龍頭"
    assert fine_industry("2317.TW","鴻海","其他電子業") == "電子中游－EMS/垂直整合"
    assert fine_industry("2308.TW","台達電","電子零組件業") == "電子中游－電源與冷卻系統"
    assert fine_industry("3017.TW","奇鋐","電腦及週邊設備業") == "電子中游－水冷散熱"
    assert _position_guidance("BULL","short")["ratio"] > _position_guidance("BEAR","short")["ratio"]


def test_balanced_evidence_v134():
    # Strong persistent company evidence should outrank a one-month / price-only
    # mirage for mid/long horizons.
    strong = {
        "monthly_revenue": {
            "available": True,
            "rows": [
                {"month": f"2026-{m:02d}", "yoy_pct": y, "mom_pct": 4.0}
                for m, y in zip(range(3, 9), [24, 29, 31, 36, 44, 53])
            ],
            "latest": {"yoy_pct": 53, "mom_pct": 10},
            "avg_yoy_3m_pct": 44.3,
        },
        "financials": {
            "available": True,
            "quarters": [
                {"eps": 8.0, "gross_margin_pct": 53.0},
                {"eps": 9.0, "gross_margin_pct": 54.0},
                {"eps": 10.0, "gross_margin_pct": 55.0},
                {"eps": 12.0, "gross_margin_pct": 56.0},
            ],
        },
        "valuation": {"available": True, "pe": 30.0},
    }
    bad = {
        "monthly_revenue": {
            "available": True,
            "rows": [
                {"month": f"2026-{m:02d}", "yoy_pct": y, "mom_pct": -10.0}
                for m, y in zip(range(3, 9), [-55, -62, -71, -80, -88, -92])
            ],
            "latest": {"yoy_pct": -92, "mom_pct": -20},
            "avg_yoy_3m_pct": -86.7,
        },
        "financials": {
            "available": True,
            "quarters": [
                {"eps": -0.4, "gross_margin_pct": -15.0},
                {"eps": -0.5, "gross_margin_pct": -25.0},
                {"eps": -0.6, "gross_margin_pct": -40.0},
                {"eps": -0.7, "gross_margin_pct": -65.0},
            ],
        },
        "valuation": {"available": False, "pe": None},
    }
    strong_long, sd = _fundamental_score(strong, "半導體業", "long")
    bad_long, bd = _fundamental_score(bad, "汽車工業", "long")
    assert strong_long is not None and bad_long is not None
    assert strong_long > bad_long + 40
    assert bd["fundamental_red_flag_penalty"] >= 30
    assert sd["revenue_positive_ratio6"] == 1.0

    # Financial-company monthly revenue can swing wildly because of accounting;
    # it must not be evaluated like manufacturing sales.
    financial = {
        "monthly_revenue": {
            "available": True,
            "rows": [{"month": "2026-08", "yoy_pct": -60, "mom_pct": 250}],
            "latest": {"yoy_pct": -60, "mom_pct": 250},
            "avg_yoy_3m_pct": -20,
        },
        "financials": {
            "available": True,
            "quarters": [{"eps": 1.5}, {"eps": 1.8}, {"eps": 2.1}, {"eps": 2.4}],
        },
        "valuation": {"available": True, "pe": 14.0},
    }
    fin_score, fd = _fundamental_score(financial, "金融保險業", "long")
    assert fin_score is not None and fin_score >= 65
    assert fd["financial_sector_method"] is True
    assert "revenue_score" not in fd

    assert _liquidity_score(10_000_000) <= 1
    assert _liquidity_score(1_000_000_000) > 60

    # Soft diversification only breaks near-ties; a materially stronger stock
    # still stays ahead even if its industry is already represented.
    stocks = []
    specs = [
        ("A", "半導體業", 90.0), ("B", "半導體業", 89.5),
        ("C", "半導體業", 88.0), ("D", "金融保險業", 87.7),
        ("E", "航運業", 87.2),
    ]
    for t, ind, sc in specs:
        stocks.append({"ticker": t, "industry": ind, "horizons": {"long": {"ranking_score": sc}}})
    picks = select_market_best({"stocks": stocks}, "long", 4)
    assert picks[0]["ticker"] == "A"
    assert "D" in {p["ticker"] for p in picks}


def test_intraday():
    hot = pd.Series({"close":108,"open":102,"high":109,"low":101,"average_price":104,"change_rate":5.1,"volume_ratio":1.8,"total_amount":1.2e9,"buy_price":107.5,"sell_price":108,"buy_volume":900,"sell_volume":500})
    f=_quote_features(hot,1.0)
    assert 0 <= f["intraday_score"] <= 100 and f["relative_market_pct_pt"]==4.1
    over = hot.copy(); over["change_rate"]=9.4; over["average_price"]=101
    g=_quote_features(over,1.0)
    assert g["overheat_penalty"] > 0 and g["state"] == "漲太快，不追"

    stocks=[]
    for i,score in enumerate([80,70,60]):
        stocks.append({"ticker":f"23{i:02d}.TW","name":str(i),"horizons":{"short":{"ranking_score":score},"mid":{"ranking_score":score},"long":{"ranking_score":score}}})
    snap={"stocks":stocks}
    q=pd.DataFrame([
        {"stock_id":"001","change_rate":1.0},
        {"stock_id":"101","change_rate":.5},
        {"stock_id":"2300","close":100,"open":98,"high":101,"low":97,"average_price":99,"change_rate":3,"volume_ratio":1.5,"total_amount":1e9,"buy_price":99.8,"sell_price":100,"buy_volume":800,"sell_volume":500},
        {"stock_id":"2301","close":100,"open":100,"high":101,"low":96,"average_price":99.5,"change_rate":.5,"volume_ratio":.8,"total_amount":3e8,"buy_price":99.8,"sell_price":100,"buy_volume":500,"sell_volume":500},
        {"stock_id":"2302","close":100,"open":99,"high":101,"low":98,"average_price":99.5,"change_rate":2,"volume_ratio":1.4,"total_amount":4e8,"buy_price":99.8,"sell_price":100,"buy_volume":600,"sell_volume":400},
    ])
    out=rerank_snapshot(snap,"short",q,top_n=3)
    assert len(out)==3 and all("live_ranking_score" in x for x in out)
    assert len(candidate_tickers(snap))==3
    tz=datetime.timezone(datetime.timedelta(hours=8))
    assert market_is_open(datetime.datetime(2026,10,2,10,0,tzinfo=tz))
    assert not market_is_open(datetime.datetime(2026,10,2,14,0,tzinfo=tz))
    assert intraday_view(out[0])["title"]




def test_validation_math():
    rows=[]
    rng=np.random.default_rng(123)
    for d in ["2024-03-29","2024-06-28","2024-09-30","2024-12-31"]:
        for i in range(100):
            score=20+i*0.7
            realized=(i-50)/1000 + rng.normal(0,0.015)
            rows.append({"date":d,"ticker":f"{1000+i}.TW","score":score,"realized_net_return":realized,"realized_alpha":realized-0.01})
    m=validation_summarize(pd.DataFrame(rows))
    assert m["cutoff_dates"]==4
    assert m["mean_cross_sectional_spearman"] is not None and m["mean_cross_sectional_spearman"] > 0
    assert m["mean_same_date_median_return_lift"] > 0



def test_universe_industry_and_etf_exclusion():
    class R:
        def __init__(self, payload, ok=True): self.payload=payload; self.ok=ok; self.status_code=200 if ok else 500
        def json(self): return self.payload
    def fake_get(url, headers=None, timeout=None, **kwargs):
        if "t187ap03_L" in url:
            return R([
                {"公司代號":"2330","公司簡稱":"台積電","產業別":"24"},
                {"公司代號":"0050","公司簡稱":"元大50","產業別":""},
            ])
        if "mopsfin_t187ap03_O" in url:
            return R([
                {"SecuritiesCompanyCode":"6488","CompanyAbbreviation":"環球晶","SecuritiesIndustryCode":"24"},
                {"SecuritiesCompanyCode":"00679","CompanyAbbreviation":"X","SecuritiesIndustryCode":""},
            ])
        raise AssertionError(url)
    old=market_data_module.requests.get
    market_data_module.requests.get=fake_get
    try:
        u=market_data_module.fetch_twse_universe()
    finally:
        market_data_module.requests.get=old
    assert set(u["ticker"]) == {"2330.TW","6488.TWO"}
    assert set(u["industry"]) == {"半導體業"}

def test_calendar_and_official_overlay():
    tz=datetime.timezone(datetime.timedelta(hours=8))
    assert calendar_reference(Path("."), now=datetime.datetime(2026,10,2,1,0,tzinfo=tz))["expected_date"]=="2026-10-01"
    assert calendar_reference(Path("."), now=datetime.datetime(2026,10,1,15,30,tzinfo=tz))["expected_date"]=="2026-09-30"
    assert calendar_reference(Path("."), now=datetime.datetime(2026,10,1,16,30,tzinfo=tz))["expected_date"]=="2026-10-01"

    class R:
        def __init__(self,p,status=200): self.p=p; self.status_code=status
        def raise_for_status(self):
            if self.status_code>=400: raise RuntimeError(self.status_code)
        def json(self): return self.p
    def fake_get(url,params=None,headers=None,timeout=None):
        if "STOCK_DAY_ALL" in url and "openapi" in url: raise RuntimeError("dns")
        if "rwd/zh/afterTrading/STOCK_DAY_ALL" in url: return R({"date":"20261001","data":[["2330","台積電","1000000","108000000","106","109","105","108","+2","10000"]]})
        if "tpex_mainboard_daily_close_quotes" in url: return R([{"Date":"1150930","SecuritiesCompanyCode":"6488","Open":"850","High":"870","Low":"845","Close":"865","TradingShares":"100000"}])
        if "/www/zh-tw/afterTrading/dailyQuotes" in url: return R({"date":"115/10/01","tables":[{"fields":["代號","名稱","收盤","漲跌","開盤","最高","最低","均價","成交股數"],"data":[["6488","環球晶","872","+7","866","878","860","871","120000"]]}]})
        if "MI_INDEX" in url and "openapi" in url: raise RuntimeError("dns")
        if "rwd/zh/afterTrading/MI_INDEX" in url: return R({"tables":[{"fields":["指數","收盤指數"],"data":[["發行量加權股價指數","48,353.49"]]}]})
        raise AssertionError(url)
    with tempfile.TemporaryDirectory() as td:
        store=DailyPriceStore(Path(td)/"p.sqlite"); old=market_data_module.requests.get; market_data_module.requests.get=fake_get
        try: meta=store.overlay_official_eod(target_date="2026-10-01")
        finally: market_data_module.requests.get=old
        assert meta["twse_fresh"] and meta["tpex_fresh"] and meta["index_fresh"]
        assert str(store.get_prices("2330.TW").index[-1].date())=="2026-10-01"



def test_antichase_and_recent_cache():
    base = synthetic_prices(314, n=500, drift=.00035, vol=.012)
    hot = base.copy()
    # Force a fresh vertical extension while preserving a realistic OHLC bar.
    prev = float(hot["Close"].iloc[-2])
    hot.iloc[-1, hot.columns.get_loc("Close")] = prev * 1.28
    hot.iloc[-1, hot.columns.get_loc("Open")] = prev * 1.08
    hot.iloc[-1, hot.columns.get_loc("High")] = prev * 1.30
    hot.iloc[-1, hot.columns.get_loc("Low")] = prev * 1.06
    plan = generate_trade_plan(hot, "short")
    assert plan is not None
    assert plan["chase_limit"] < float(hot["Close"].iloc[-1])
    assert evaluate_entry_state(hot, plan) == "DO_NOT_CHASE"

    settings = SimpleNamespace(commission=.001425,sell_tax=.003,slippage=.0005)
    bench = synthetic_prices(315, n=500, drift=.0002, vol=.01)
    f = estimate_horizon_return(hot,"short",settings,benchmark_df=bench)
    stock={"ticker":"9999.TW","name":"HOT","avg_volume_20d":2e6,"research":{},"horizons":{"short":{"forecast":f,"plan":plan,"entry_state":"DO_NOT_CHASE"}}}
    _combined_ranking_score(stock,"short","price_only")
    assert stock["horizons"]["short"]["score_components"]["entry_penalty"] >= 10.0

    with tempfile.TemporaryDirectory() as td:
        store=DailyPriceStore(Path(td)/"p.sqlite")
        rows=[]
        for dt,row in base.iloc[:200].iterrows():
            rows.append(("2330.TW",dt.strftime("%Y-%m-%d"),float(row.Open),float(row.High),float(row.Low),float(row.Close),float(row.Volume)))
        import sqlite3
        with sqlite3.connect(store.db_path) as conn:
            conn.executemany("INSERT OR REPLACE INTO daily_prices(ticker,date,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",rows)
        recent=store.get_prices("2330.TW",limit=140)
        assert len(recent)==140
        assert recent.index[-1] == base.iloc[:200].index[-1]

def main():
    test_model_core(); test_100_series_stress(); test_research_and_copy(); test_balanced_evidence_v134(); test_intraday(); test_validation_math(); test_universe_industry_and_etf_exclusion(); test_calendar_and_official_overlay(); test_antichase_and_recent_cache()
    print("Alpha Radar V13.4 offline self-test: PASS")

if __name__ == "__main__": main()
