"""Trading-date freshness helpers for Taiwan Alpha Radar V13."""
from __future__ import annotations

import datetime
import json
from pathlib import Path


def _closure_days(data_dir: Path) -> set[str]:
    path = Path(data_dir) / "closure_notices.json"
    if not path.exists():
        return set()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {str(x.get("closure_day")) for x in rows if isinstance(x, dict) and x.get("closure_day")}
    except Exception:
        return set()


def _previous_session(day: datetime.date, closures: set[str], include_day: bool = True) -> datetime.date:
    d = day if include_day else day - datetime.timedelta(days=1)
    for _ in range(14):
        if d.weekday() < 5 and d.isoformat() not in closures:
            return d
        d -= datetime.timedelta(days=1)
    return d


def calendar_reference(data_dir: Path, now: datetime.datetime = None, allow_fetch: bool = False) -> dict:
    tz = datetime.timezone(datetime.timedelta(hours=8))
    now = now or datetime.datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)
    closures = _closure_days(Path(data_dir))
    today = now.date()
    today_closed = today.weekday() >= 5 or today.isoformat() in closures

    # Official TWSE/TPEx full-market EOD files are normally published after the
    # close, often around 16:00. Before 16:30 we keep the previous complete
    # session as the freshness target instead of declaring today's data stale.
    after_eod_buffer = (now.hour, now.minute) >= (16, 30)
    if today_closed:
        expected = _previous_session(today, closures, include_day=False)
    elif after_eod_buffer:
        expected = today
    else:
        expected = _previous_session(today, closures, include_day=False)

    return {
        "status": "VALID",
        "today": today.isoformat(),
        "expected_date": expected.isoformat(),
        "is_weekend": today.weekday() >= 5,
        "is_known_closure": today.isoformat() in closures,
        "after_eod_buffer": after_eod_buffer,
        "operator_notice_count": len(closures),
        "warnings": [],
    }


def daily_freshness(price_date: str, calendar: dict) -> dict:
    expected = str(calendar.get("expected_date") or "")
    current = bool(price_date and expected and str(price_date) >= expected)
    return {
        "current_daily": current,
        "status": "FRESH" if current else "STALE",
        "expected_date": expected,
        "price_date": str(price_date or ""),
    }


def entry_review_allowed(price_date: str, plan: dict | None, calendar: dict) -> bool:
    return bool(plan is not None and daily_freshness(price_date, calendar).get("current_daily"))


def save_closure_notice(data_dir: Path, closure_day: str, url: str, at_time: str, now: datetime.datetime = None, confirmed: bool = False):
    if not confirmed:
        raise ValueError("必須核對並勾選確認官方公告。")
    datetime.date.fromisoformat(closure_day)  # validation
    notice_file = Path(data_dir) / "closure_notices.json"
    notices = []
    if notice_file.exists():
        try:
            notices = json.loads(notice_file.read_text(encoding="utf-8"))
        except Exception:
            notices = []
    notices = [x for x in notices if x.get("closure_day") != closure_day]
    notices.append({"closure_day": closure_day, "url": url, "at_time": at_time})
    notice_file.write_text(json.dumps(notices, ensure_ascii=False, indent=2), encoding="utf-8")
