#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Загрузка последних N свечей USDT-M фьючерсов Binance в формате NumPy.

РЕЖИМЫ:
- include_incomplete_last=True  → включать текущую незакрытую свечу (если она есть).
- include_incomplete_last=False → исключать её ТОЛЬКО если она «только началась»
  (age <= just_started_cutoff), иначе оставлять. Всегда возвращаем ровно `limit` строк
  (при достаточной истории).

Параметры порога «только началась»:
- just_started_cutoff_ratio: доля длительности интервала (по умолчанию 0.05 = 5%).
- just_started_cutoff_min_ms: минимальный порог в мс (по умолчанию 2000 мс).

Формат выхода: np.ndarray формы (M, 6), строки:
    [timestamp_ms, open, high, low, close, volume]  (dtype=float64)

Примеры:
    # 60 последних 1h свечей, включая текущую (по умолчанию)
    data_inc = get_futures_klines_np("BTCUSDT", "1h", 60)

    # 60 последних 1h свечей: если 10:01, свеча 10:00 «только началась» — отрежем;
    # если 09:58, свеча 09:00 идёт давно — оставим.
    data_exc = get_futures_klines_np("BTCUSDT", "1h", 60, include_incomplete_last=False)
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
import numpy as np
from datetime import datetime, timezone, timedelta

BINANCE_FAPI_BASE = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
TIME_ENDPOINT = "/fapi/v1/time"

_ALLOWED_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
}

_CHUNK_LIMIT = 1000  # максимум за один запрос


class BinanceAPIError(RuntimeError):
    pass


def _http_get(session: requests.Session, url: str, params: Dict[str, Any],
              retries: int = 3, timeout: float = 15.0):
    """Надёжный GET с экспоненциальным бэкоффом для 429/5xx."""
    backoff = 0.5
    for _ in range(retries):
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 4.0)
            continue
        try:
            payload = resp.json()
        except Exception:
            payload = {"message": resp.text}
        raise BinanceAPIError(
            f"HTTP {resp.status_code}: {payload.get('code','?')} "
            f"{payload.get('msg') or payload.get('message')}"
        )
    raise BinanceAPIError("Exceeded retry limit while calling Binance API.")


def _get_server_time_ms(session: requests.Session) -> int:
    """Текущее серверное время Binance в миллисекундах."""
    data = _http_get(session, BINANCE_FAPI_BASE + TIME_ENDPOINT, params={})
    server_ms = int(data.get("serverTime"))
    return server_ms


def _normalise_interval(interval: str) -> str:
    """
    Привести ввод к каноническому формату Binance.
    Поддерживаем алиасы месяца: '1mo','1mon','1month' → '1M'.
    """
    if not isinstance(interval, str) or not interval.strip():
        raise ValueError("`interval` must be a non-empty string.")
    raw = interval.strip()
    low = raw.lower()

    if raw == "1M" or low in {"1mo", "1mon", "1month"}:
        return "1M"

    candidates = {
        "1m","3m","5m","15m","30m",
        "1h","2h","4h","6h","8h","12h",
        "1d","3d","1w"
    }
    if low in candidates:
        return low

    if low == "1m" and raw == "1M":
        return "1M"

    raise ValueError(f"Unsupported interval '{interval}'. Allowed: {sorted(_ALLOWED_INTERVALS)}")


def _interval_ms_static(interval: str) -> Optional[int]:
    """
    Длительность интервала в мс для фиксированных ТФ; для '1M' → None.
    """
    table = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 3_600_000,
        "2h": 2 * 3_600_000,
        "4h": 4 * 3_600_000,
        "6h": 6 * 3_600_000,
        "8h": 8 * 3_600_000,
        "12h": 12 * 3_600_000,
        "1d": 86_400_000,
        "3d": 3 * 86_400_000,
        "1w": 7 * 86_400_000,
        "1M": None,
    }
    return table[interval]


def _month_bounds(dt: datetime) -> Tuple[datetime, datetime]:
    """Начало и начало следующего месяца (UTC)."""
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if dt.month == 12:
        nxt = start.replace(year=dt.year + 1, month=1)
    else:
        nxt = start.replace(month=dt.month + 1)
    return start, nxt


def _current_open_time_ms(server_ms: int, interval: str) -> int:
    """
    Открытие ТЕКУЩЕЙ свечи (UTC, мс) для данного интервала.
    """
    dt = datetime.fromtimestamp(server_ms / 1000, tz=timezone.utc)

    if interval == "1M":
        first, _ = _month_bounds(dt)
        return int(first.timestamp() * 1000)

    if interval == "1w":
        monday = dt - timedelta(days=dt.weekday(),
                                hours=dt.hour,
                                minutes=dt.minute,
                                seconds=dt.second,
                                microseconds=dt.microsecond)
        return int(monday.timestamp() * 1000)

    if interval == "1d":
        day0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return int(day0.timestamp() * 1000)

    if interval == "3d":
        base = int(datetime(1970, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        T = _interval_ms_static("3d")
        return ((server_ms - base) // T) * T + base

    # Минутные/часовые
    T = _interval_ms_static(interval)
    return (server_ms // T) * T


def _interval_len_ms_dynamic(server_ms: int, interval: str) -> int:
    """
    Длительность интервала (мс), динамически для '1M', статически для остальных.
    """
    T = _interval_ms_static(interval)
    if T is not None:
        return T
    # '1M': длина текущего месяца
    dt = datetime.fromtimestamp(server_ms / 1000, tz=timezone.utc)
    start, nxt = _month_bounds(dt)
    return int((nxt - start).total_seconds() * 1000)


def get_futures_klines_np(
    symbol: str,
    interval: str,
    limit: int,
    include_incomplete_last: bool = True,
    *,
    just_started_cutoff_ratio: float = 0.05,
    just_started_cutoff_min_ms: int = 2000,
) -> np.ndarray:
    """
    Вернёт последние `limit` свечей USDT-M по `symbol`/`interval`.

    Логика последней свечи:
      - include_incomplete_last=True  → включаем текущую незакрытую свечу.
      - include_incomplete_last=False → исключаем ТОЛЬКО если age <= cutoff,
        где cutoff = max(just_started_cutoff_min_ms, round(ratio * interval_len_ms)).
        Иначе — оставляем её (даже если она формально незакрыта).

    Возвращаем ровно `limit` строк (при достаточной истории). Порядок: старые → новые.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("`symbol` must be a non-empty string like 'BTCUSDT'.")
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("`limit` must be a positive integer.")
    if not (0.0 < just_started_cutoff_ratio <= 0.5):
        raise ValueError("`just_started_cutoff_ratio` must be in (0, 0.5].")
    if just_started_cutoff_min_ms < 0:
        raise ValueError("`just_started_cutoff_min_ms` must be >= 0.")

    symbol = symbol.strip().upper()
    interval = _normalise_interval(interval)

    url = BINANCE_FAPI_BASE + KLINES_ENDPOINT
    rows: List[List[float]] = []
    remaining = limit

    with requests.Session() as s:
        end_time: Optional[int] = None

        # Определяем, нужно ли отрезать текущую свечу (только если она «только началась»).
        if not include_incomplete_last:
            server_ms = _get_server_time_ms(s)
            open_ms = _current_open_time_ms(server_ms, interval)
            age_ms = max(0, server_ms - open_ms)

            interval_len_ms = _interval_len_ms_dynamic(server_ms, interval)
            cutoff_ms = max(just_started_cutoff_min_ms,
                            int(round(just_started_cutoff_ratio * interval_len_ms)))

            # Если свеча «свежая» — исключаем её (endTime до начала текущей)
            if age_ms <= cutoff_ms:
                end_time = open_ms - 1
            else:
                end_time = None  # оставляем текущую свечу (даже незакрытую)

        while remaining > 0:
            batch_limit = min(_CHUNK_LIMIT, remaining)
            params: Dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "limit": batch_limit,
            }
            if end_time is not None:
                params["endTime"] = end_time

            data = _http_get(s, url, params=params)
            if not isinstance(data, list) or not data:
                break

            batch_rows = [
                [
                    float(d[0]),  # open time ms
                    float(d[1]),  # open
                    float(d[2]),  # high
                    float(d[3]),  # low
                    float(d[4]),  # close
                    float(d[5]),  # base volume
                ]
                for d in data
            ]

            # Binance отдаёт в пачке по возрастанию времени.
            rows = batch_rows + rows
            remaining -= len(batch_rows)

            earliest_open_time = int(data[0][0])
            end_time = earliest_open_time - 1  # двигаем окно в прошлое

            time.sleep(0.05)  # уважаем лимиты

    if not rows:
        return np.empty((0, 6), dtype=np.float64)

    if len(rows) > limit:
        rows = rows[-limit:]

    return np.asarray(rows, dtype=np.float64)
