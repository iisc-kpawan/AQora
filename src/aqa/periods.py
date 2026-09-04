from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta

from .errors import ValidationError


def resolve_period(period: str, available_end: datetime, now: datetime | None = None) -> tuple[datetime, datetime]:
    """Resolve a period phrase to a (start, end) range.

    "last month"/"last N <unit>"/"latest available" describe a reporting
    period relative to the newest data this dataset actually has, so they
    anchor to `available_end`. "today"/"yesterday"/"this week"/"this month"/
    "this year" name a literal calendar period, so they anchor to `now` (the
    caller's current time in the dataset's own local timezone) instead —
    if that calendar period isn't covered by the data, the caller's own
    available-range clamp turns this into an honest "not available yet"
    result rather than silently substituting an older period.
    """
    text = period.strip().casefold()
    now = now or datetime.now()
    start_of_now_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text in {"last month", "latest month", "latest available"}:
        start = available_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, available_end
    if text in {"today", "current day"}:
        return start_of_now_day, now
    if text == "yesterday":
        yesterday = start_of_now_day - timedelta(days=1)
        return yesterday, yesterday.replace(hour=23, minute=59, second=59)
    if text in {"this week", "current week"}:
        return start_of_now_day - timedelta(days=start_of_now_day.weekday()), now
    if text in {"this month", "current month"}:
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), now
    if text in {
        "this year", "current year", "year to date", "ytd",
        "start of year", "start of the year", "beginning of year", "beginning of the year",
    }:
        return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0), now
    match = re.fullmatch(r"(?:last|past)\s+(?:(\d+)\s+)?(?:complete\s+)?(hours?|days?|weeks?|months?|years?)", text)
    if match:
        count, unit = int(match.group(1)) if match.group(1) else 1, match.group(2)
        if count > 1000:
            raise ValidationError("Relative period is too large")
        if unit.startswith("hour"):
            return available_end - timedelta(hours=count - 1), available_end
        if unit.startswith("day"):
            return available_end - timedelta(days=count) + timedelta(hours=1), available_end
        if unit.startswith("week"):
            return available_end - timedelta(weeks=count) + timedelta(hours=1), available_end
        if unit.startswith("month"):
            end = available_end
            if "complete" in text:
                end = available_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(seconds=1)
            start = end
            for _ in range(count - 1):
                start = (start.replace(day=1) - timedelta(days=1)).replace(day=1)
            return start.replace(hour=0, minute=0, second=0, microsecond=0), end
        return available_end.replace(year=available_end.year - count, month=1, day=1, hour=0, minute=0, second=0, microsecond=0), available_end
    for fmt in ("%Y-%m", "%Y"):
        try:
            start = datetime.strptime(text, fmt)
            if fmt == "%Y-%m":
                end = start.replace(day=calendar.monthrange(start.year, start.month)[1], hour=23, minute=59, second=59)
            else:
                end = start.replace(month=12, day=31, hour=23, minute=59, second=59)
            return start, min(end, available_end)
        except ValueError:
            pass
    try:
        start = datetime.strptime(text, "%B %Y")
        end = start.replace(day=calendar.monthrange(start.year, start.month)[1], hour=23, minute=59, second=59)
        return start, min(end, available_end)
    except ValueError as exc:
        raise ValidationError(f"Unsupported period: {period}") from exc
