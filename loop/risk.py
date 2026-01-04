from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class RiskRegimeResult:
    risk_on: bool
    vol_high: bool
    regime: str  # "risk-on / low-vol" etc.

    # диагностические значения (все "на момент открытия дня входа")
    close_last: float
    sma_last: float
    vol_last: float
    vol_threshold: float

    warmup: bool
    reason: str


def calc_risk_regime_for_next_day_open(
    daily_hist_ohlcv: np.ndarray,
    *,
    sma_days: int = 200,
    vol_days: int = 30,
    vol_median_window: int = 365,
) -> RiskRegimeResult:
    """
    Режим НА ОТКРЫТИИ "следующего дня" (того дня, в который вы входите intraday).

    Вход: daily_hist_ohlcv содержит только полностью закрытые дневные свечи.
    Последняя строка массива = вчерашний день.

    Как считается:
    - risk_on: вчерашнее закрытие сравнивается со средней ценой закрытия за последние sma_days дней
    - vol: берём дневные доходности (закрытие делим на открытие, затем минус один) за последние vol_days дней
           и считаем разброс этих доходностей
    - vol_threshold: медиана прошлых значений vol (по окну vol_median_window) и сравнение текущего vol с порогом
    """
    a = np.asarray(daily_hist_ohlcv, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 6:
        raise ValueError("daily_hist_ohlcv must be 2D ndarray with shape (N, 6).")

    n = a.shape[0]
    if n < max(sma_days, vol_days) + 1:
        return RiskRegimeResult(
            risk_on=False, vol_high=False, regime="warmup",
            close_last=float("nan"), sma_last=float("nan"),
            vol_last=float("nan"), vol_threshold=float("nan"),
            warmup=True,
            reason=f"Need at least {max(sma_days, vol_days) + 1} full daily candles in history."
        )

    # сортировка по времени на всякий случай
    order = np.argsort(a[:, 0])
    a = a[order]

    opens = a[:, 1]
    closes = a[:, 4]

    # дневные доходности для истории
    with np.errstate(divide="ignore", invalid="ignore"):
        day_ret = (closes / opens) - 1.0

    # ---- Trend (risk-on/off) ----
    close_last = float(closes[-1])
    sma_window = closes[-sma_days:]
    sma_last = float(np.mean(sma_window))
    risk_on = bool(close_last > sma_last)

    # ---- Vol (high/low) ----
    vol_window = day_ret[-vol_days:]
    vol_last = float(np.std(vol_window, ddof=0))

    # ---- Threshold for vol_high: median of past vol values ----
    # считаем "исторические vol" по прошлым дням, чтобы сравнение было стабильным
    # и чтобы порог не зависел от одного текущего значения
    vol_series = []
    for j in range(vol_days, n):
        w = day_ret[j - vol_days : j]
        vol_series.append(float(np.std(w, ddof=0)))

    if len(vol_series) < 20:
        return RiskRegimeResult(
            risk_on=risk_on, vol_high=False, regime="warmup",
            close_last=close_last, sma_last=sma_last,
            vol_last=vol_last, vol_threshold=float("nan"),
            warmup=True,
            reason="Not enough vol history to build a stable threshold."
        )

    # порог берём по прошлым значениям, исключая самое последнее (чтобы порог не подстраивался под текущую волу)
    use = vol_series[:-1] if len(vol_series) > 1 else vol_series
    if len(use) > vol_median_window:
        use = use[-vol_median_window:]
    vol_threshold = float(np.median(np.asarray(use, dtype=np.float64)))

    vol_high = bool(vol_last > vol_threshold)

    # ---- Regime label ----
    if risk_on and (not vol_high):
        regime = "risk-on / low-vol"
    elif risk_on and vol_high:
        regime = "risk-on / high-vol"
    elif (not risk_on) and (not vol_high):
        regime = "risk-off / low-vol"
    else:
        regime = "risk-off / high-vol"

    return RiskRegimeResult(
        risk_on=risk_on,
        vol_high=vol_high,
        regime=regime,
        close_last=close_last,
        sma_last=sma_last,
        vol_last=vol_last,
        vol_threshold=vol_threshold,
        warmup=False,
        reason="OK"
    )
