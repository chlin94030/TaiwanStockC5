"""
Taiwan Alpha Radar V13.3 - market + research data layer.

Price source: Yahoo Finance, with local SQLite cache.
Research source: FinMind v4 (monthly revenue, quarterly financials, PER,
institutional flows). Broker-branch/main-force proxy is optional because the
underlying branch dataset is a Sponsor feature and "main force" has no official
single definition.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
try:
    import yfinance as yf
except ImportError:  # surfaced cleanly at runtime; dependency is in requirements.txt
    yf = None


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_BRANCH_URL = "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report"


def _taipei_timestamp() -> datetime.datetime:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    return datetime.datetime.now(tz)


def _finmind_token() -> str:
    return os.getenv("FINMIND_TOKEN", "").strip()


def provider_runtime_status(store=None) -> dict:
    return {
        "sqlite_connected": True,
        "yfinance_ready": yf is not None,
        "finmind_token_configured": bool(_finmind_token()),
        "branch_flow_enabled": os.getenv("ENABLE_BRANCH_FLOW", "0") == "1",
        "timestamp": _taipei_timestamp().strftime("%Y-%m-%d %H:%M:%S"),
    }


def stock_id_from_ticker(ticker: str) -> str:
    return str(ticker or "").split(".")[0].strip()


def _yf_history_args(period: str) -> dict:
    """Translate UI periods to yfinance-supported arguments.

    yfinance does not support arbitrary period strings such as 3y/8y, so those
    are converted to an explicit start date instead of silently failing.
    """
    valid = {"1y", "2y", "5y", "10y", "max"}
    p = str(period or "5y")
    if p in valid:
        return {"period": p}
    if p.endswith("y") and p[:-1].isdigit():
        years = max(1, int(p[:-1]))
        start = (_taipei_timestamp().date() - datetime.timedelta(days=years * 366 + 10)).isoformat()
        return {"start": start}
    return {"period": "5y"}


INDUSTRY_CODE_LABELS = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維",
    "05": "電機機械", "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙工業",
    "10": "鋼鐵工業", "11": "橡膠工業", "12": "汽車工業", "14": "建材營造",
    "15": "航運業", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨",
    "19": "綜合", "20": "其他", "21": "化學工業", "22": "生技醫療業",
    "23": "油電燃氣業", "24": "半導體業", "25": "電腦及週邊設備業", "26": "光電業",
    "27": "通信網路業", "28": "電子零組件業", "29": "電子通路業", "30": "資訊服務業",
    "31": "其他電子業", "32": "文化創意業", "33": "農業科技業", "35": "綠能環保",
    "36": "數位雲端", "37": "運動休閒", "38": "居家生活", "80": "管理股票",
}


def _industry_label(value) -> str:
    raw = str(value or "").strip().replace("　", "")
    if not raw:
        return ""
    if raw in INDUSTRY_CODE_LABELS:
        return INDUSTRY_CODE_LABELS[raw]
    # Some sources occasionally return an integer-like code without zero padding.
    if raw.isdigit() and len(raw) == 1 and f"0{raw}" in INDUSTRY_CODE_LABELS:
        return INDUSTRY_CODE_LABELS[f"0{raw}"]
    # If an upstream provider already supplies a name, retain it.
    return raw


def fetch_twse_universe() -> pd.DataFrame:
    """Fetch listed + OTC common-stock universe with official broad industry."""
    tickers: list[tuple[str, str, str]] = []
    headers = {"User-Agent": "Mozilla/5.0 AlphaRadar/13.3"}

    # TWSE company master is preferred because it carries the official industry
    # code. STOCK_DAY_ALL remains a resilient fallback if the master is down.
    twse_loaded = False
    try:
        r = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", headers=headers, timeout=10)
        if r.ok:
            for item in r.json():
                code = str(item.get("公司代號") or item.get("Code") or "").strip()
                name = str(item.get("公司簡稱") or item.get("公司名稱") or item.get("Name") or "").strip()
                industry = _industry_label(item.get("產業別") or item.get("IndustryCode") or "")
                if len(code) == 4 and code.isdigit() and not code.startswith("0"):
                    tickers.append((f"{code}.TW", name, industry or "上市"))
                    twse_loaded = True
    except Exception:
        pass

    if not twse_loaded:
        try:
            r = requests.get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL", headers=headers, timeout=10)
            if r.ok:
                for item in r.json():
                    code = str(item.get("Code", "")).strip()
                    name = str(item.get("Name", "")).strip()
                    if len(code) == 4 and code.isdigit() and not code.startswith("0"):
                        tickers.append((f"{code}.TW", name, "上市"))
        except Exception:
            pass

    try:
        r = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", headers=headers, timeout=10)
        if not r.ok:
            r = requests.get("https://www.tpex.org.tw/openapi/v1/mopsfront_t187ap03_O", headers=headers, timeout=10)
        if r.ok:
            for item in r.json():
                code = str(
                    item.get("SecuritiesCompanyCode")
                    or item.get("公司代號")
                    or item.get("公司代碼")
                    or ""
                ).strip()
                name = str(
                    item.get("CompanyAbbreviation")
                    or item.get("CompanyName")
                    or item.get("Company Name")
                    or item.get("公司簡稱")
                    or item.get("公司名稱")
                    or ""
                ).strip()
                industry = _industry_label(
                    item.get("SecuritiesIndustryCode") or item.get("產業別") or item.get("IndustryCode") or ""
                )
                if len(code) == 4 and code.isdigit() and not code.startswith("0"):
                    tickers.append((f"{code}.TWO", name, industry or "上櫃"))
    except Exception:
        pass

    if not tickers:
        backup = [
            ("2330.TW", "台積電", "半導體業"), ("2317.TW", "鴻海", "其他電子業"),
            ("2454.TW", "聯發科", "半導體業"), ("2382.TW", "廣達", "電腦及週邊設備業"),
            ("3231.TW", "緯創", "電腦及週邊設備業"), ("3017.TW", "奇鋐", "電腦及週邊設備業"),
            ("6669.TW", "緯穎", "電腦及週邊設備業"), ("1519.TW", "華城", "電機機械"),
        ]
        return pd.DataFrame(backup, columns=["ticker", "name", "industry"])

    return pd.DataFrame(tickers, columns=["ticker", "name", "industry"]).drop_duplicates("ticker")


class DailyPriceStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_cache_error: str | None = None
        self.download_errors: list[str] = []
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS daily_prices (
                    ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
                    PRIMARY KEY (ticker, date)
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS store_meta (
                    meta_key TEXT PRIMARY KEY, value TEXT
                )"""
            )

    def clear(self) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM daily_prices")
                conn.execute("DELETE FROM store_meta")
            return True
        except Exception as exc:
            self.last_cache_error = str(exc)
            return False

    def _meta_get(self, key: str, default: str = "") -> str:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute("SELECT value FROM store_meta WHERE meta_key=?", (key,)).fetchone()
            return str(row[0]) if row else default
        except Exception:
            return default

    def _meta_set(self, key: str, value: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT OR REPLACE INTO store_meta(meta_key,value) VALUES(?,?)", (key, str(value)))
        except Exception:
            pass

    @staticmethod
    def _period_years(period: str) -> int:
        p = str(period or "5y").lower()
        if p == "max":
            return 99
        if p.endswith("y") and p[:-1].isdigit():
            return max(1, int(p[:-1]))
        return 5

    def _coverage_counts(self, tickers: list[str]) -> dict[str, int]:
        if not tickers:
            return {}
        try:
            # Query once without a giant IN (...) placeholder list; some SQLite
            # builds cap bound variables near 999 while the Taiwan universe is
            # >2,000 symbols. Filter the small aggregate result in Python.
            wanted = set(map(str, tickers))
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT ticker, COUNT(*) FROM daily_prices GROUP BY ticker"
                ).fetchall()
            return {str(t): int(n) for t, n in rows if str(t) in wanted}
        except Exception:
            return {}

    @staticmethod
    def _ticker_frame(data: pd.DataFrame, ticker: str, chunk_len: int) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()
        try:
            if isinstance(data.columns, pd.MultiIndex):
                lvl0 = set(map(str, data.columns.get_level_values(0)))
                lvl1 = set(map(str, data.columns.get_level_values(1)))
                if ticker in lvl0:
                    out = data[ticker].copy()
                elif ticker in lvl1:
                    out = data.xs(ticker, axis=1, level=1).copy()
                else:
                    return pd.DataFrame()
            else:
                out = data.copy() if chunk_len == 1 else pd.DataFrame()
            wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
            return out[wanted].dropna(how="all") if wanted else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def batch_fetch_and_update(
        self,
        tickers: list[str],
        period: str = "5y",
        chunk_size: int = 80,
        progress=None,
        force_full: bool = False,
        full_refresh_days: int = 7,
    ) -> dict:
        """Refresh price history efficiently.

        Cold start / weekly maintenance downloads the requested history.  Normal
        daily updates reuse the local multi-year history and only backfill new or
        insufficient tickers; the official TWSE/TPEx overlay then supplies the
        newest complete session.  This preserves the model input while avoiding
        downloading ~2,000 five-year histories on every button click.
        """
        if yf is None:
            raise RuntimeError("缺少 yfinance；請先執行 pip install -r requirements.txt")
        unique = list(dict.fromkeys([t for t in tickers if t]))
        self.download_errors = []
        if not unique:
            return {"downloaded_tickers": 0, "errors": [], "mode": "none", "requested": 0}

        coverage = self._coverage_counts(unique)
        cached_period = self._period_years(self._meta_get("history_period", "0y"))
        requested_period = self._period_years(period)
        last_full_text = self._meta_get("last_full_refresh", "")
        last_full = None
        try:
            last_full = datetime.datetime.fromisoformat(last_full_text).date() if last_full_text else None
        except Exception:
            last_full = None
        age_days = 999 if last_full is None else (_taipei_timestamp().date() - last_full).days
        covered_ratio = sum(1 for t in unique if coverage.get(t, 0) >= 140) / max(1, len(unique))
        full_refresh = bool(
            force_full
            or cached_period < requested_period
            or age_days >= int(full_refresh_days)
            or covered_ratio < 0.70
        )

        if full_refresh:
            fetch_list = unique
            fetch_period = period
            mode = "full"
        else:
            # Newly listed / newly discovered / failed tickers still get history.
            fetch_list = [t for t in unique if coverage.get(t, 0) < 140]
            fetch_period = period
            mode = "incremental"

        if progress:
            progress(0, max(1, len(fetch_list)), "完整歷史更新" if full_refresh else "沿用歷史快取")

        downloaded = 0
        total_chunks = max(1, (len(fetch_list) + chunk_size - 1) // chunk_size) if fetch_list else 0
        for i in range(0, len(fetch_list), chunk_size):
            chunk = fetch_list[i:i + chunk_size]
            chunk_no = i // chunk_size + 1
            if progress:
                progress(i, len(fetch_list), f"行情批次 {chunk_no}/{total_chunks}（{min(i+len(chunk),len(fetch_list))}/{len(fetch_list)} 檔）")
            data = None
            exc_last = None
            for attempt in range(2):
                try:
                    history_args = _yf_history_args(fetch_period)
                    data = yf.download(
                        chunk,
                        **history_args,
                        group_by="ticker",
                        auto_adjust=True,
                        actions=False,
                        progress=False,
                        threads=True,
                        timeout=20,
                    )
                    break
                except Exception as exc:
                    exc_last = exc
                    if progress:
                        progress(i, len(fetch_list), f"行情批次 {chunk_no}/{total_chunks} 重試 {attempt+1}/2")
                    time.sleep(0.6 * (attempt + 1))
            if data is None or getattr(data, "empty", True):
                self.download_errors.append(f"chunk {chunk_no}: {exc_last or 'empty response'}")
                continue

            records: list[tuple] = []
            for ticker in chunk:
                frame = self._ticker_frame(data, ticker, len(chunk))
                if frame.empty:
                    continue
                frame = frame.copy()
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                count_before = len(records)
                for idx, row in frame.iterrows():
                    try:
                        close = float(row.get("Close", np.nan))
                        if not np.isfinite(close) or close <= 0:
                            continue
                        op = float(row.get("Open", close)); hi = float(row.get("High", close)); lo = float(row.get("Low", close))
                        vol = float(row.get("Volume", 0.0))
                        records.append((ticker, idx.strftime("%Y-%m-%d"), op, hi, lo, close, vol))
                    except Exception:
                        continue
                if len(records) > count_before:
                    downloaded += 1

            if records:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        conn.executemany(
                            """INSERT OR REPLACE INTO daily_prices
                            (ticker, date, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""", records
                        )
                except Exception as exc:
                    self.download_errors.append(f"sqlite write: {exc}")
            if progress:
                progress(min(i + len(chunk), len(fetch_list)), len(fetch_list), f"行情批次 {chunk_no}/{total_chunks} 完成")

        if full_refresh and downloaded >= max(5, int(len(fetch_list) * 0.70)):
            self._meta_set("history_period", period)
            self._meta_set("last_full_refresh", _taipei_timestamp().isoformat(timespec="seconds"))

        if progress and not fetch_list:
            progress(1, 1, "歷史資料已在快取，直接更新最新交易日")
        return {
            "downloaded_tickers": downloaded,
            "errors": self.download_errors[:],
            "mode": mode,
            "requested": len(fetch_list),
            "covered_ratio": round(covered_ratio, 3),
            "last_full_refresh": last_full_text,
        }

    @staticmethod
    def _num(value):
        try:
            text = str(value).replace(",", "").strip()
            if text in {"", "--", "---", "X", "-"}:
                return None
            v = float(text)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    @staticmethod
    def _roc_date(value):
        text = str(value or "").strip().replace("/", "").replace("-", "")
        if len(text) == 7 and text.isdigit():
            return f"{int(text[:3]) + 1911:04d}-{text[3:5]}-{text[5:7]}"
        if len(text) == 8 and text.isdigit():
            return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
        return ""

    def overlay_official_eod(self, target_date: str | None = None, progress=None) -> dict:
        """Overlay the newest official TWSE/TPEx EOD bars.

        Source order is deliberately redundant because cloud hosts sometimes
        reach Yahoo but fail one of the Taiwan exchange OpenAPI domains:
          1) official latest-snapshot OpenAPI
          2) official date-specific TWSE/TPEx web endpoint for ``target_date``

        The function returns source/date metadata so the UI can show exactly
        what was updated instead of silently presenting a stale Yahoo session.
        """
        records: list[tuple] = []
        counts = {
            "twse": 0, "tpex": 0, "index": 0, "errors": [],
            "target_date": str(target_date or ""),
            "twse_date": "", "tpex_date": "", "index_date": "",
            "twse_source": "", "tpex_source": "", "index_source": "",
            "twse_fresh": False, "tpex_fresh": False, "index_fresh": False,
            "twse_target_response": False, "twse_target_has_data": False,
            "tpex_target_response": False, "tpex_target_has_data": False,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AlphaRadar/13.2; +https://github.com/)",
            "Accept": "application/json,text/plain,*/*",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        def get_json(url: str, *, params: dict | None = None, label: str = "", extra_headers: dict | None = None):
            try:
                req_headers = dict(headers)
                if extra_headers:
                    req_headers.update(extra_headers)
                r = requests.get(url, params=params, headers=req_headers, timeout=18)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                counts["errors"].append(f"{label or url}: {type(exc).__name__}: {exc}")
                return None

        def newest_date(rows: list[tuple], suffix: str) -> str:
            ds = [str(x[1]) for x in rows if str(x[0]).endswith(suffix) and x[1]]
            return max(ds) if ds else ""

        # ---------- TWSE: latest snapshot ----------
        twse_rows: list[tuple] = []
        payload = get_json(
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
            label="TWSE OpenAPI latest",
        )
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("Code", "")).strip()
                if len(code) != 4 or not code.isdigit():
                    continue
                date = self._roc_date(row.get("Date"))
                op = self._num(row.get("OpeningPrice")); hi = self._num(row.get("HighestPrice"))
                lo = self._num(row.get("LowestPrice")); cl = self._num(row.get("ClosingPrice"))
                vol = self._num(row.get("TradeVolume"))
                if date and cl is not None and cl > 0:
                    twse_rows.append((f"{code}.TW", date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if twse_rows:
                counts["twse_source"] = "TWSE OpenAPI"

        twse_date = newest_date(twse_rows, ".TW")

        # ---------- TWSE fallback: date-specific RWD ----------
        # Important for Streamlit/GitHub cloud environments where the OpenAPI
        # hostname may be blocked or where Yahoo is one session behind.
        if target_date and (not twse_date or twse_date < target_date):
            ymd = str(target_date).replace("-", "")
            fb = get_json(
                "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY_ALL",
                params={"response": "json", "date": ymd},
                label=f"TWSE RWD {target_date}",
            )
            fb_rows: list[tuple] = []
            if isinstance(fb, dict):
                counts["twse_target_response"] = True
                fb_date = self._roc_date(fb.get("date")) or str(target_date)
                data_rows = fb.get("data") if isinstance(fb.get("data"), list) else []
                # RWD STOCK_DAY_ALL row layout:
                # code,name,volume,value,open,high,low,close,change,transactions
                for row in data_rows:
                    if not isinstance(row, (list, tuple)) or len(row) < 8:
                        continue
                    code = str(row[0]).strip()
                    if len(code) != 4 or not code.isdigit():
                        continue
                    op, hi, lo, cl = map(self._num, [row[4], row[5], row[6], row[7]])
                    vol = self._num(row[2])
                    if cl is not None and cl > 0:
                        fb_rows.append((f"{code}.TW", fb_date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if fb_rows:
                counts["twse_target_has_data"] = True
                twse_rows.extend(fb_rows)
                twse_date = newest_date(fb_rows, ".TW")
                counts["twse_source"] = "TWSE RWD date-specific"

        records.extend(twse_rows)
        counts["twse"] = len({r[0] for r in twse_rows if r[1] == (twse_date or r[1])}) if twse_rows else 0
        counts["twse_date"] = twse_date
        counts["twse_fresh"] = bool(twse_date and (not target_date or twse_date >= target_date))
        if progress:
            progress(1, 3, f"官方行情：上市 {twse_date or '未取得'}")

        # ---------- TPEx: latest snapshot ----------
        tpex_rows: list[tuple] = []
        payload = get_json(
            "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
            label="TPEx OpenAPI latest",
        )
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                code = str(row.get("SecuritiesCompanyCode") or row.get("Code") or "").strip()
                if len(code) != 4 or not code.isdigit():
                    continue
                date = self._roc_date(row.get("Date"))
                op = self._num(row.get("Open") or row.get("OpeningPrice"))
                hi = self._num(row.get("High") or row.get("HighestPrice"))
                lo = self._num(row.get("Low") or row.get("LowestPrice"))
                cl = self._num(row.get("Close") or row.get("ClosingPrice"))
                vol = self._num(row.get("TradingShares") or row.get("TradeVolume"))
                if date and cl is not None and cl > 0:
                    tpex_rows.append((f"{code}.TWO", date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if tpex_rows:
                counts["tpex_source"] = "TPEx OpenAPI"

        tpex_date = newest_date(tpex_rows, ".TWO")

        # ---------- TPEx fallback: date-specific official dailyQuotes ----------
        if target_date and (not tpex_date or tpex_date < target_date):
            tpex_referer = {"Referer": "https://www.tpex.org.tw/zh-tw/mainboard/trading/info/pricing.html"}
            fb = get_json(
                "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
                params={"response": "json", "date": str(target_date).replace("-", "/")},
                label=f"TPEx dailyQuotes {target_date}",
                extra_headers=tpex_referer,
            )
            # Some TPEx deployments expose the same JSON through the older
            # o=json query form. Retry it only when the first variant did not
            # return a table, keeping both paths official.
            if not (isinstance(fb, dict) and isinstance(fb.get("tables"), list) and fb.get("tables")):
                fb2 = get_json(
                    "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes",
                    params={"l": "zh-tw", "s": "0,asc,0", "o": "json", "date": str(target_date).replace("-", "/")},
                    label=f"TPEx dailyQuotes alt {target_date}",
                    extra_headers=tpex_referer,
                )
                if isinstance(fb2, dict):
                    fb = fb2
            fb_rows: list[tuple] = []
            if isinstance(fb, dict):
                counts["tpex_target_response"] = True
                fb_date = self._roc_date(fb.get("date") or fb.get("Date")) or str(target_date)
                tables = fb.get("tables") if isinstance(fb.get("tables"), list) else []
                for table in tables:
                    if not isinstance(table, dict):
                        continue
                    fields = [str(x).replace(" ", "").strip() for x in (table.get("fields") or [])]
                    data_rows = table.get("data") if isinstance(table.get("data"), list) else []
                    if not data_rows:
                        continue

                    def idx(names: tuple[str, ...], default: int | None = None):
                        for name in names:
                            if name in fields:
                                return fields.index(name)
                        return default

                    # The current official table order is code,name,close,change,
                    # open,high,low,avg,volume,...; field-based lookup is preferred.
                    i_code = idx(("代號", "證券代號"), 0)
                    i_close = idx(("收盤", "收盤價"), 2)
                    i_open = idx(("開盤", "開盤價"), 4)
                    i_high = idx(("最高", "最高價"), 5)
                    i_low = idx(("最低", "最低價"), 6)
                    i_vol = idx(("成交股數", "成交量"), 8)
                    for row in data_rows:
                        if not isinstance(row, (list, tuple)):
                            continue
                        need = max(x for x in [i_code, i_close, i_open, i_high, i_low, i_vol] if x is not None)
                        if len(row) <= need:
                            continue
                        code = str(row[i_code]).strip() if i_code is not None else ""
                        if len(code) != 4 or not code.isdigit():
                            continue
                        cl = self._num(row[i_close]) if i_close is not None else None
                        op = self._num(row[i_open]) if i_open is not None else None
                        hi = self._num(row[i_high]) if i_high is not None else None
                        lo = self._num(row[i_low]) if i_low is not None else None
                        vol = self._num(row[i_vol]) if i_vol is not None else None
                        if cl is not None and cl > 0:
                            fb_rows.append((f"{code}.TWO", fb_date, op or cl, hi or cl, lo or cl, cl, vol or 0.0))
            if fb_rows:
                counts["tpex_target_has_data"] = True
                tpex_rows.extend(fb_rows)
                tpex_date = newest_date(fb_rows, ".TWO")
                counts["tpex_source"] = "TPEx dailyQuotes date-specific"

        records.extend(tpex_rows)
        counts["tpex"] = len({r[0] for r in tpex_rows if r[1] == (tpex_date or r[1])}) if tpex_rows else 0
        counts["tpex_date"] = tpex_date
        counts["tpex_fresh"] = bool(tpex_date and (not target_date or tpex_date >= target_date))
        if progress:
            progress(2, 3, f"官方行情：上櫃 {tpex_date or '未取得'}")

        # ---------- TAIEX close: OpenAPI then date-specific RWD fallback ----------
        index_row = None
        payload = get_json(
            "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
            label="TWSE index OpenAPI latest",
        )
        if isinstance(payload, list):
            for row in payload:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("指數") or row.get("Index") or row.get("Name") or "")
                if "發行量加權股價指數" not in name and "TAIEX" not in name.upper():
                    continue
                date = self._roc_date(row.get("日期") or row.get("Date"))
                cl = self._num(row.get("收盤指數") or row.get("ClosingIndex") or row.get("Close"))
                if date and cl is not None and cl > 0:
                    index_row = ("^TWII", date, cl, cl, cl, cl, 0.0)
                    counts["index_source"] = "TWSE index OpenAPI"
                    break

        if target_date and (index_row is None or str(index_row[1]) < target_date):
            ymd = str(target_date).replace("-", "")
            fb = get_json(
                "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX",
                params={"response": "json", "date": ymd, "type": "ALLBUT0999"},
                label=f"TWSE index RWD {target_date}",
            )
            if isinstance(fb, dict):
                tables = fb.get("tables") if isinstance(fb.get("tables"), list) else []
                for table in tables:
                    if not isinstance(table, dict):
                        continue
                    fields = [str(x).strip() for x in (table.get("fields") or [])]
                    data_rows = table.get("data") if isinstance(table.get("data"), list) else []
                    if not fields or not data_rows:
                        continue
                    try:
                        name_i = next(i for i, x in enumerate(fields) if "指數" in x and "收盤" not in x)
                        close_i = next(i for i, x in enumerate(fields) if "收盤指數" in x)
                    except StopIteration:
                        continue
                    for row in data_rows:
                        if not isinstance(row, (list, tuple)) or len(row) <= max(name_i, close_i):
                            continue
                        name = str(row[name_i])
                        if "發行量加權股價指數" not in name:
                            continue
                        cl = self._num(row[close_i])
                        if cl is not None and cl > 0:
                            index_row = ("^TWII", str(target_date), cl, cl, cl, cl, 0.0)
                            counts["index_source"] = "TWSE index RWD date-specific"
                            break
                    if index_row is not None and str(index_row[1]) >= str(target_date):
                        break

        if index_row is not None:
            records.append(index_row)
            counts["index"] = 1
            counts["index_date"] = str(index_row[1])
            counts["index_fresh"] = bool(not target_date or str(index_row[1]) >= target_date)
        if progress:
            progress(3, 3, f"官方行情：大盤 {counts.get('index_date') or '未取得'}")

        if records:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.executemany(
                        """INSERT OR REPLACE INTO daily_prices
                        (ticker, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        records,
                    )
            except Exception as exc:
                counts["errors"].append(f"official EOD sqlite: {type(exc).__name__}: {exc}")

        counts["all_markets_fresh"] = bool(
            counts["twse_fresh"] and counts["tpex_fresh"] and counts["index_fresh"]
        )
        return counts

    def get_prices(self, ticker: str, limit: int | None = None) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            if limit is not None and int(limit) > 0:
                df = pd.read_sql_query(
                    "SELECT date, open as Open, high as High, low as Low, close as Close, volume as Volume "
                    "FROM (SELECT * FROM daily_prices WHERE ticker = ? ORDER BY date DESC LIMIT ?) "
                    "ORDER BY date ASC",
                    conn, params=(ticker, int(limit)),
                )
            else:
                df = pd.read_sql_query(
                    "SELECT date, open as Open, high as High, low as Low, close as Close, volume as Volume "
                    "FROM daily_prices WHERE ticker = ? ORDER BY date ASC",
                    conn, params=(ticker,),
                )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.drop_duplicates("date").set_index("date").sort_index()
        return df


class ResearchDataClient:
    """Cached research-data client for fields shown next to recommendations."""

    def __init__(self, db_path: Path, token: str | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.token = (token if token is not None else _finmind_token()).strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 AlphaRadar/13.0"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self._init_cache()

    def _init_cache(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS research_cache (
                    cache_key TEXT PRIMARY KEY,
                    fetched_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )"""
            )

    def _cache_get(self, key: str, ttl_hours: float) -> Any | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT fetched_at, payload FROM research_cache WHERE cache_key=?", (key,)
                ).fetchone()
            if not row:
                return None
            ts = datetime.datetime.fromisoformat(row[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            age = datetime.datetime.now(datetime.timezone.utc) - ts.astimezone(datetime.timezone.utc)
            if age.total_seconds() > ttl_hours * 3600:
                return None
            return json.loads(row[1])
        except Exception:
            return None

    def _cache_set(self, key: str, payload: Any):
        try:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO research_cache(cache_key,fetched_at,payload) VALUES(?,?,?)",
                    (key, now, text),
                )
        except Exception:
            pass

    def _finmind(self, dataset: str, stock_id: str, start_date: str, end_date: str | None = None, ttl_hours: float = 6.0) -> list[dict]:
        key = f"fm:{dataset}:{stock_id}:{start_date}:{end_date or ''}"
        cached = self._cache_get(key, ttl_hours)
        if isinstance(cached, list):
            return cached
        params = {"dataset": dataset, "data_id": stock_id, "start_date": start_date}
        if end_date:
            params["end_date"] = end_date
        if self.token:
            params["token"] = self.token
        try:
            r = self.session.get(FINMIND_URL, params=params, timeout=12)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and js.get("status") not in (None, 200, "200"):
                return []
            data = js.get("data", []) if isinstance(js, dict) else []
            if isinstance(data, list):
                self._cache_set(key, data)
                return data
        except Exception:
            pass
        return []

    @staticmethod
    def _date_ago(days: int) -> str:
        return (_taipei_timestamp().date() - datetime.timedelta(days=days)).isoformat()

    def monthly_revenue(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockMonthRevenue", stock_id, self._date_ago(800), ttl_hours=18)
        if not rows:
            return {"available": False, "rows": []}
        df = pd.DataFrame(rows)
        if df.empty or "revenue" not in df.columns:
            return {"available": False, "rows": []}
        for c in ["revenue", "revenue_year", "revenue_month"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["revenue", "revenue_year", "revenue_month"])
        if df.empty:
            return {"available": False, "rows": []}
        df["period"] = pd.to_datetime(
            df["revenue_year"].astype(int).astype(str) + "-" + df["revenue_month"].astype(int).astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )
        df = df.dropna(subset=["period"]).sort_values("period").drop_duplicates("period", keep="last")
        df["mom"] = df["revenue"].pct_change()
        period_map = {p.to_period("M"): r for p, r in zip(df["period"], df["revenue"])}
        df["yoy"] = [
            (rev / period_map.get(p.to_period("M") - 12) - 1.0)
            if period_map.get(p.to_period("M") - 12) not in (None, 0) else np.nan
            for p, rev in zip(df["period"], df["revenue"])
        ]
        tail = df.tail(12)
        out_rows = [
            {
                "month": r.period.strftime("%Y-%m"),
                "revenue_billion": round(float(r.revenue) / 1e8, 2),  # 億元 = 1e8 NTD
                "mom_pct": round(float(r.mom) * 100.0, 2) if pd.notna(r.mom) else None,
                "yoy_pct": round(float(r.yoy) * 100.0, 2) if pd.notna(r.yoy) else None,
            }
            for r in tail.itertuples()
        ]
        latest = out_rows[-1] if out_rows else {}
        yoy_vals = [r["yoy_pct"] for r in out_rows[-3:] if r.get("yoy_pct") is not None]
        return {
            "available": bool(out_rows),
            "latest": latest,
            "rows": out_rows,
            "avg_yoy_3m_pct": round(float(np.mean(yoy_vals)), 2) if yoy_vals else None,
        }

    def financials(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockFinancialStatements", stock_id, self._date_ago(650), ttl_hours=48)
        if not rows:
            return {"available": False, "quarters": []}
        df = pd.DataFrame(rows)
        if df.empty or not {"date", "type", "value"}.issubset(df.columns):
            return {"available": False, "quarters": []}
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date", "value", "type"])
        if df.empty:
            return {"available": False, "quarters": []}
        p = df.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()
        if p.empty:
            return {"available": False, "quarters": []}

        rev_col = next((c for c in ["Revenue", "OperatingRevenue", "TotalRevenue", "Income"] if c in p.columns), None)
        gp_col = "GrossProfit" if "GrossProfit" in p.columns else None
        eps_col = "EPS" if "EPS" in p.columns else None

        q = p.tail(4)
        quarters = []
        for dt, row in q.iterrows():
            eps = float(row[eps_col]) if eps_col and pd.notna(row.get(eps_col)) else None
            rev = float(row[rev_col]) if rev_col and pd.notna(row.get(rev_col)) else None
            gp = float(row[gp_col]) if gp_col and pd.notna(row.get(gp_col)) else None
            gm = (gp / rev * 100.0) if gp is not None and rev not in (None, 0) else None
            quarters.append({
                "quarter": f"{dt.year}Q{((dt.month - 1) // 3) + 1}",
                "date": dt.strftime("%Y-%m-%d"),
                "eps": round(eps, 2) if eps is not None else None,
                "gross_margin_pct": round(gm, 2) if gm is not None and np.isfinite(gm) else None,
            })

        # YTD running sum of FinMind single-quarter EPS.  Corporate actions can
        # cause the arithmetic sum to differ from the issuer-reported cumulative EPS,
        # so the UI labels this as an unadjusted sum.
        running_by_year = {}
        for item in quarters:
            year = int(item["quarter"][:4])
            if item.get("eps") is not None:
                running_by_year[year] = running_by_year.get(year, 0.0) + float(item["eps"])
                item["eps_ytd_sum_unadjusted"] = round(running_by_year[year], 2)
            else:
                item["eps_ytd_sum_unadjusted"] = None

        eps_vals = [x["eps"] for x in quarters if x.get("eps") is not None]
        gm_vals = [x["gross_margin_pct"] for x in quarters if x.get("gross_margin_pct") is not None]
        return {
            "available": bool(quarters),
            "quarters": quarters,
            "eps_4q_sum": round(float(np.sum(eps_vals)), 2) if eps_vals else None,
            "gross_margin_4q_avg_pct": round(float(np.mean(gm_vals)), 2) if gm_vals else None,
            "gross_margin_latest_pct": gm_vals[-1] if gm_vals else None,
        }

    def valuation(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockPER", stock_id, self._date_ago(50), ttl_hours=6)
        if not rows:
            return {"available": False}
        df = pd.DataFrame(rows)
        if df.empty:
            return {"available": False}
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
        row = df.iloc[-1]
        pe = pd.to_numeric(pd.Series([row.get("PER")]), errors="coerce").iloc[0]
        pbr = pd.to_numeric(pd.Series([row.get("PBR")]), errors="coerce").iloc[0]
        yld = pd.to_numeric(pd.Series([row.get("dividend_yield")]), errors="coerce").iloc[0]
        return {
            "available": bool(pd.notna(pe) or pd.notna(pbr)),
            "date": str(row.get("date", ""))[:10],
            "pe": round(float(pe), 2) if pd.notna(pe) else None,
            "pbr": round(float(pbr), 2) if pd.notna(pbr) else None,
            "dividend_yield_pct": round(float(yld), 2) if pd.notna(yld) else None,
        }

    def institutional_flow(self, stock_id: str) -> dict:
        rows = self._finmind("TaiwanStockInstitutionalInvestorsBuySellWide", stock_id, self._date_ago(45), ttl_hours=2)
        if not rows:
            # Compatibility fallback to long table.
            long_rows = self._finmind("TaiwanStockInstitutionalInvestorsBuySell", stock_id, self._date_ago(45), ttl_hours=2)
            return self._institutional_long(long_rows)
        df = pd.DataFrame(rows)
        if df.empty:
            return {"available": False}
        for c in df.columns:
            if c.endswith("_buy") or c.endswith("_sell"):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").tail(22)

        def net(buy, sell):
            if buy in df.columns and sell in df.columns:
                return float((df[buy] - df[sell]).sum())
            return 0.0

        foreign = net("Foreign_Investor_buy", "Foreign_Investor_sell") + net("Foreign_Dealer_Self_buy", "Foreign_Dealer_Self_sell")
        trust = net("Investment_Trust_buy", "Investment_Trust_sell")
        if "Dealer_buy" in df.columns and "Dealer_sell" in df.columns:
            dealer = net("Dealer_buy", "Dealer_sell")
        else:
            dealer = net("Dealer_self_buy", "Dealer_self_sell") + net("Dealer_Hedging_buy", "Dealer_Hedging_sell")
        total = foreign + trust + dealer
        return {
            "available": True,
            "sessions": int(len(df)),
            "foreign_net_lots": round(foreign / 1000.0, 1),
            "trust_net_lots": round(trust / 1000.0, 1),
            "dealer_net_lots": round(dealer / 1000.0, 1),
            "total_net_lots": round(total / 1000.0, 1),
            "summary": self._flow_text(total / 1000.0, foreign / 1000.0, trust / 1000.0, dealer / 1000.0),
        }

    def _institutional_long(self, rows: list[dict]) -> dict:
        if not rows:
            return {"available": False}
        df = pd.DataFrame(rows)
        if df.empty or not {"buy", "sell", "name"}.issubset(df.columns):
            return {"available": False}
        df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0.0)
        df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0.0)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            latest_dates = sorted(df["date"].dropna().unique())[-22:]
            df = df[df["date"].isin(latest_dates)]
        nets = df.assign(net=df["buy"] - df["sell"]).groupby("name")["net"].sum()
        foreign = float(nets[[i for i in nets.index if "Foreign" in str(i) or "外資" in str(i)]].sum())
        trust = float(nets[[i for i in nets.index if "Investment_Trust" in str(i) or "投信" in str(i)]].sum())
        dealer = float(nets[[i for i in nets.index if "Dealer" in str(i) or "自營" in str(i)]].sum())
        total = foreign + trust + dealer
        return {
            "available": True,
            "sessions": int(df["date"].nunique()) if "date" in df.columns else None,
            "foreign_net_lots": round(foreign / 1000.0, 1),
            "trust_net_lots": round(trust / 1000.0, 1),
            "dealer_net_lots": round(dealer / 1000.0, 1),
            "total_net_lots": round(total / 1000.0, 1),
            "summary": self._flow_text(total / 1000.0, foreign / 1000.0, trust / 1000.0, dealer / 1000.0),
        }

    @staticmethod
    def _flow_text(total, foreign, trust, dealer) -> str:
        direction = "偏買超" if total > 0 else "偏賣超" if total < 0 else "大致中性"
        return f"近月三大法人{direction} {total:+,.0f} 張（外資 {foreign:+,.0f}、投信 {trust:+,.0f}、自營商 {dealer:+,.0f}）"

    def branch_main_force_proxy(self, stock_id: str, max_sessions: int = 20) -> dict:
        """
        Optional broker-branch concentration proxy.

        Disabled by default.  It is intentionally labelled as a proxy because
        'main force' is not an official investor category.  FinMind's branch data
        is Sponsor-only and one trading date is requested at a time.
        """
        if os.getenv("ENABLE_BRANCH_FLOW", "0") != "1":
            return {
                "available": False,
                "reason": "DISABLED",
                "summary": "未啟用券商分點資料；主力屬代理指標，不以三大法人冒充。",
            }
        if not self.token:
            return {
                "available": False,
                "reason": "TOKEN_REQUIRED",
                "summary": "主力代理需 FinMind Sponsor 分點權限；目前未設定 FINMIND_TOKEN。",
            }

        end = _taipei_timestamp().date()
        days = []
        d = end
        while len(days) < max_sessions and (end - d).days < 20:
            if d.weekday() < 5:
                days.append(d)
            d -= datetime.timedelta(days=1)

        frames = []
        for day in days:
            key = f"branch:{stock_id}:{day.isoformat()}"
            cached = self._cache_get(key, 4)
            if isinstance(cached, list):
                rows = cached
            else:
                params = {"data_id": stock_id, "start_date": day.isoformat(), "token": self.token}
                try:
                    r = self.session.get(FINMIND_BRANCH_URL, params=params, timeout=10)
                    r.raise_for_status()
                    js = r.json()
                    rows = js.get("data", []) if isinstance(js, dict) else []
                    if isinstance(rows, list):
                        self._cache_set(key, rows)
                except Exception:
                    rows = []
            if rows:
                frames.append(pd.DataFrame(rows))

        if not frames:
            return {
                "available": False,
                "reason": "NO_BRANCH_DATA",
                "summary": "分點資料目前無可用回傳（可能為權限、休市或資料尚未更新）。",
            }
        df = pd.concat(frames, ignore_index=True)
        if not {"securities_trader", "buy", "sell"}.issubset(df.columns):
            return {"available": False, "reason": "SCHEMA_MISMATCH", "summary": "分點資料欄位格式不符。"}
        df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0.0)
        df["sell"] = pd.to_numeric(df["sell"], errors="coerce").fillna(0.0)
        by_branch = df.assign(net=df["buy"] - df["sell"]).groupby("securities_trader")["net"].sum().sort_values()
        top_buy = by_branch.tail(5).sort_values(ascending=False)
        top_sell = by_branch.head(5)
        # Concentration proxy: top five net buys + top five net sells, expressed separately.
        buy_lots = float(top_buy.clip(lower=0).sum()) / 1000.0
        sell_lots = float((-top_sell.clip(upper=0)).sum()) / 1000.0
        proxy = buy_lots - sell_lots
        return {
            "available": True,
            "sessions": len(frames),
            "proxy_net_lots": round(proxy, 1),
            "top_buy_lots": round(buy_lots, 1),
            "top_sell_lots": round(sell_lots, 1),
            "top_buy_branches": [{"name": str(k), "net_lots": round(float(v) / 1000.0, 1)} for k, v in top_buy.items()],
            "top_sell_branches": [{"name": str(k), "net_lots": round(float(v) / 1000.0, 1)} for k, v in top_sell.items()],
            "summary": f"近 {len(frames)} 個交易日分點代理：前5大淨買與前5大淨賣差額 {proxy:+,.0f} 張（非官方『主力』分類）",
        }

    def stock_research(self, ticker: str, include_branch: bool = False) -> dict:
        stock_id = stock_id_from_ticker(ticker)
        rev = self.monthly_revenue(stock_id)
        fin = self.financials(stock_id)
        val = self.valuation(stock_id)
        inst = self.institutional_flow(stock_id)
        branch = self.branch_main_force_proxy(stock_id) if include_branch else {
            "available": False,
            "reason": "NOT_REQUESTED",
            "summary": "主力分點代理未啟用。",
        }
        return {
            "stock_id": stock_id,
            "monthly_revenue": rev,
            "financials": fin,
            "valuation": val,
            "institutional_flow": inst,
            "main_force_proxy": branch,
            "source": "FinMind v4 + optional sponsor branch data",
            "fetched_at": _taipei_timestamp().isoformat(timespec="seconds"),
        }
