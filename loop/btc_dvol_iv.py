#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
helpers/dvol_iv.py

Оценка IV для короткой экспирации (по умолчанию 26 часов) на основе Deribit DVOL (30-day annualised IV).

DVOL по Deribit — это 30-дневная (constant 30-DTE) годовая (annualised) implied volatility оценка. :contentReference[oaicite:1]{index=1}

Что делает модуль:
- Один раз загружает CSV BTC_DVOL_3600s_20200101_20251212.csv из корня проекта и кеширует.
- По timestamp_ms (мс) выбирает "снимок" DVOL без утечки:
  * если timestamp попадает внутрь часа — берём OPEN этого часа (значение на старте часа);
  * если exact часа в файле нет — берём последний доступный час <= timestamp (safe).
- Пересчитывает 30d IV -> IV на expiry_hours через mean-reversion по variance:
  * каузально оцениваем AR(1) на variance (v_t = a + b v_{t-1}) по скользящему окну, только по прошлым данным;
  * из b получаем k = -ln(b) (per-hour);
  * из a,b получаем долгосрочную v_bar = a/(1-b);
  * интерпретируем DVOL^2 как среднюю ожидаемую variance на горизонте 30d (v_avg_30d),
    инвертируем для "spot-like" v0, затем считаем v_avg на expiry_hours.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None


SnapshotMode = Literal["open", "prev_close"]


@dataclass(frozen=True)
class DVOLCache:
    path: str
    ts_ms: np.ndarray               # shape (n,), int64, start-of-hour timestamps
    dvol_open: np.ndarray           # shape (n,), float64, DVOL in %
    dvol_close: np.ndarray          # shape (n,), float64, DVOL in %
    v_snap: np.ndarray              # shape (n,), float64, annualised variance from chosen snapshot (decimal^2)
    k_per_hour: np.ndarray          # shape (n,), float64
    vbar: np.ndarray                # shape (n,), float64


_DVOL_CACHE: Optional[DVOLCache] = None


def _project_root_default_csv() -> str:
    # helpers/.. -> project root
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    return os.path.join(root, "/home/jupiter/PYTHON/MARKET_DATA/BTC_DVOL_3600s_20210101_20260106.csv")


def _ffill_np(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaN in numpy array (in-place safe copy)."""
    x = np.asarray(x, dtype=np.float64).copy()
    isn = ~np.isfinite(x)
    if not isn.any():
        return x
    # индексы последних валидных
    idx = np.arange(x.size)
    idx[isn] = -1
    np.maximum.accumulate(idx, out=idx)
    # если первые элементы NaN, idx будет -1 -> заполним их ближайшим первым валидным (если есть)
    first_valid = np.where(idx >= 0)[0]
    if first_valid.size == 0:
        return np.full_like(x, np.nan, dtype=np.float64)
    fv = first_valid[0]
    x[:fv] = x[fv]
    x[isn] = x[idx[isn]]
    return x


def _load_dvol_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает (ts_ms, open, close) отсортированные по ts_ms.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DVOL CSV not found: {path}")

    if pd is not None:
        df = pd.read_csv(
            path,
            usecols=["timestamp_ms", "open", "close"],
            dtype={"timestamp_ms": "int64"},
        )
        df = df.dropna(subset=["timestamp_ms"]).copy()
        df = df.sort_values("timestamp_ms")
        # дедуп: если вдруг есть повтор timestamp_ms, оставляем последний
        df = df.drop_duplicates(subset=["timestamp_ms"], keep="last")

        ts = df["timestamp_ms"].to_numpy(dtype=np.int64, copy=True)
        op = df["open"].to_numpy(dtype=np.float64, copy=True)
        cl = df["close"].to_numpy(dtype=np.float64, copy=True)
    else:
        import csv
        ts_list, op_list, cl_list = [], [], []
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    ts_list.append(int(row["timestamp_ms"]))
                except Exception:
                    continue
                def _to_f(v):
                    try:
                        return float(v)
                    except Exception:
                        return np.nan
                op_list.append(_to_f(row.get("open", "")))
                cl_list.append(_to_f(row.get("close", "")))

        ts = np.asarray(ts_list, dtype=np.int64)
        op = np.asarray(op_list, dtype=np.float64)
        cl = np.asarray(cl_list, dtype=np.float64)

        order = np.argsort(ts)
        ts, op, cl = ts[order], op[order], cl[order]
        # дедуп
        if ts.size > 1:
            keep = np.ones(ts.size, dtype=bool)
            keep[:-1] = ts[:-1] != ts[1:]
            ts, op, cl = ts[keep], op[keep], cl[keep]

    # ffill, чтобы не было дыр по NaN (лучше иметь последнюю известную оценку, чем NaN)
    op = _ffill_np(op)
    cl = _ffill_np(cl)

    return ts, op, cl


def _precompute_causal_ar1_params(
    v: np.ndarray,
    lookback_pairs: int,
    *,
    min_pairs: int = 48,
    b_min: float = 0.01,
    b_max: float = 0.999,
    half_life_hours_fallback: float = 7.0 * 24.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Каузально оцениваем AR(1) на variance:
      v_t = a + b * v_{t-1} + eps
    по скользящему окну длиной lookback_pairs (по парам).

    Возвращает:
      k_per_hour[t], vbar[t]
    где k = -ln(b), vbar = a/(1-b)

    Важно: для каждого t используются только данные <= t (нет утечки).
    """
    v = np.asarray(v, dtype=np.float64)
    n = int(v.size)

    if n <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    if n < 3:
        k = np.full(n, np.nan, dtype=np.float64)
        vb = np.full(n, np.nan, dtype=np.float64)
        return k, vb

    # пары (x_k, y_k) = (v_k, v_{k+1}), k=0..n-2
    x = v[:-1]
    y = v[1:]
    m = n - 1

    cs_x = np.concatenate(([0.0], np.cumsum(x)))
    cs_y = np.concatenate(([0.0], np.cumsum(y)))
    cs_x2 = np.concatenate(([0.0], np.cumsum(x * x)))
    cs_xy = np.concatenate(([0.0], np.cumsum(x * y)))

    # fallback k (per-hour) из half-life
    k_fb = float(np.log(2.0) / max(1e-9, half_life_hours_fallback))

    # кумулятивная средняя variance как запасной vbar (ДОЛЖНА БЫТЬ длины n)
    cs_v = np.concatenate(([0.0], np.cumsum(v)))  # len = n+1
    cum_mean_v = cs_v[1:] / np.arange(1, n + 1, dtype=np.float64)  # len = n

    k_out = np.full(n, np.nan, dtype=np.float64)
    vbar_out = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        # последний доступный pair index, который включает v_t: это k_end = t-1
        k_end = t - 1
        if k_end < 0:
            # t=0: нет пары
            k_out[t] = k_fb
            vbar_out[t] = max(1e-12, float(cum_mean_v[t]))
            continue

        k_start = max(0, k_end - int(lookback_pairs) + 1)
        cnt = k_end - k_start + 1
        if cnt < int(min_pairs):
            k_out[t] = k_fb
            vbar_out[t] = max(1e-12, float(cum_mean_v[t]))
            continue

        # window sums over pairs [k_start .. k_end]
        sum_x = cs_x[k_end + 1] - cs_x[k_start]
        sum_y = cs_y[k_end + 1] - cs_y[k_start]
        sum_x2 = cs_x2[k_end + 1] - cs_x2[k_start]
        sum_xy = cs_xy[k_end + 1] - cs_xy[k_start]

        mean_x = sum_x / cnt
        mean_y = sum_y / cnt
        var_x = (sum_x2 / cnt) - (mean_x * mean_x)
        cov_xy = (sum_xy / cnt) - (mean_x * mean_y)

        if not np.isfinite(var_x) or var_x <= 1e-18 or not np.isfinite(cov_xy):
            k_out[t] = k_fb
            vbar_out[t] = max(1e-12, float(cum_mean_v[t]))
            continue

        b = cov_xy / var_x
        if not np.isfinite(b):
            k_out[t] = k_fb
            vbar_out[t] = max(1e-12, float(cum_mean_v[t]))
            continue

        b = float(np.clip(b, b_min, b_max))
        a = float(mean_y - b * mean_x)

        denom = 1.0 - b
        if denom <= 1e-9:
            vbar = float(mean_y)
            k = 0.0
        else:
            vbar = float(a / denom)
            vbar = max(1e-12, vbar)
            k = float(-np.log(b))  # per-hour

        if not np.isfinite(k) or k < 0.0:
            k = k_fb
        if not np.isfinite(vbar) or vbar <= 0.0:
            vbar = max(1e-12, float(cum_mean_v[t]))

        k_out[t] = k
        vbar_out[t] = vbar

    return k_out, vbar_out



def _ensure_cache_loaded(
    csv_path: Optional[str],
    *,
    lookback_hours: int,
    snapshot_mode: SnapshotMode,
) -> DVOLCache:
    global _DVOL_CACHE

    path = csv_path or _project_root_default_csv()

    if _DVOL_CACHE is not None and os.path.abspath(_DVOL_CACHE.path) == os.path.abspath(path):
        return _DVOL_CACHE

    ts, op, cl = _load_dvol_csv(path)

    # DVOL в процентах -> в долях
    op_dec = np.maximum(op, 1e-12) / 100.0
    cl_dec = np.maximum(cl, 1e-12) / 100.0

    # snapshot variance series (annualised variance)
    if snapshot_mode == "open":
        sigma = op_dec
    elif snapshot_mode == "prev_close":
        # prev_close для t: берём close предыдущего часа (для t=0 fallback на open)
        sigma = cl_dec.copy()
        sigma[1:] = cl_dec[:-1]
        sigma[0] = op_dec[0]
    else:
        raise ValueError(f"Unknown snapshot_mode: {snapshot_mode}")

    v = np.square(sigma)  # annualised variance

    # каузальная оценка mean-reversion параметров по окну lookback_hours (в парах)
    k_per_hour, vbar = _precompute_causal_ar1_params(
        v,
        lookback_pairs=int(max(24, lookback_hours)),
    )

    _DVOL_CACHE = DVOLCache(
        path=path,
        ts_ms=ts,
        dvol_open=op,
        dvol_close=cl,
        v_snap=v,
        k_per_hour=k_per_hour,
        vbar=vbar,
    )
    return _DVOL_CACHE


def get_iv_from_dvol(
    timestamp_ms: int,
    *,
    csv_path: Optional[str] = None,
    expiry_hours: float = 26.0,
    ref_days: float = 30.0,
    lookback_hours: int = 24 * 60,          # 60 дней истории для каузальной калибровки AR(1)
    snapshot_mode: SnapshotMode = "open",   # "open" = значение на старте часа (safe)
    output: Literal["dec", "pct"] = "dec",
    clamp_iv: tuple[float, float] = (0.01, 5.00),  # 1%..500% в долях
) -> Optional[float]:
    """
    Вернуть оценку ATM IV (annualised) для опциона с экспирацией через expiry_hours,
    используя DVOL 30-day annualised (constant 30-DTE) как вход.

    timestamp_ms: время "сейчас" в миллисекундах.

    Без утечки:
    - timestamp приводится к началу часа (floor),
    - берётся последний доступный ts <= floor_ts,
    - используются только каузальные параметры к этому моменту.

    output:
      - "dec": IV как доля (0.85)
      - "pct": IV как процент (85.0)
    """
    try:
        ts_q = int(timestamp_ms)
    except Exception:
        return None
    if ts_q <= 0:
        return None

    cache = _ensure_cache_loaded(
        csv_path,
        lookback_hours=lookback_hours,
        snapshot_mode=snapshot_mode,
    )

    # приводим к началу часа
    HOUR_MS = 3_600_000
    ts_floor = ts_q - (ts_q % HOUR_MS)

    ts_arr = cache.ts_ms
    idx = int(np.searchsorted(ts_arr, ts_floor, side="right") - 1)
    if idx < 0 or idx >= ts_arr.size:
        return None

    # v_avg_ref: "30d average variance" proxy (annualised variance)
    v_avg_ref = float(cache.v_snap[idx])

    # параметры mean reversion (per-hour)
    k = float(cache.k_per_hour[idx])
    vbar = float(cache.vbar[idx])

    # пересчёт горизонтов
    T_ref_h = float(ref_days) * 24.0
    T_h = float(expiry_hours)

    # safety
    if not np.isfinite(v_avg_ref) or v_avg_ref <= 0.0:
        return None
    if not np.isfinite(k) or k < 0.0:
        k = 0.0
    if not np.isfinite(vbar) or vbar <= 0.0:
        vbar = v_avg_ref

    # Если k≈0, term structure плоская
    if k <= 1e-10:
        sigma_T = float(np.sqrt(max(v_avg_ref, 1e-18)))
    else:
        # v_avg(T) = vbar + (v0 - vbar)*(1 - e^{-kT})/(kT)
        # => v0 = vbar + (v_avg - vbar) * (kT)/(1 - e^{-kT})
        denom_ref = 1.0 - float(np.exp(-k * T_ref_h))
        if denom_ref <= 1e-12:
            v0 = v_avg_ref
        else:
            v0 = vbar + (v_avg_ref - vbar) * (k * T_ref_h) / denom_ref

        # v_avg на целевом горизонте
        denom_T = k * T_h
        if denom_T <= 1e-12:
            v_avg_T = v0
        else:
            coef = (1.0 - float(np.exp(-k * T_h))) / denom_T
            v_avg_T = vbar + (v0 - vbar) * coef

        sigma_T = float(np.sqrt(max(v_avg_T, 1e-18)))

    # clamp
    lo, hi = float(clamp_iv[0]), float(clamp_iv[1])
    sigma_T = float(np.clip(sigma_T, lo, hi))

    if output == "pct":
        return sigma_T * 100.0
    return sigma_T


def get_iv_26h(timestamp_ms: int, *, csv_path: Optional[str] = None) -> Optional[float]:
    """
    Упрощённая обёртка: вернуть IV (в долях) для 26ч опциона.
    """
    return get_iv_from_dvol(
        timestamp_ms,
        csv_path=csv_path,
        expiry_hours=26.0,
        ref_days=30.0,
        snapshot_mode="open",
        output="dec",
    )



