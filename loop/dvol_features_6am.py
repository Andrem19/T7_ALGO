# -*- coding: utf-8 -*-
"""
helpers/dvol_features_6am.py

Фичи на момент 06:00 UTC (ts_ms) из:
- исторических часовых DVOL свечей (dict ts_ms -> {open,high,low,close})
- исторических 1h свечей BTC (список/np-массив строк, как у Binance klines)

Главная функция:
    calc_features_at_ts(ts_ms, ctx) -> dict

Рекомендуемое использование:
    ctx = FeatureContext(btc_1h=sv.data_1h, dvol_1h=sv.data_dvol)
    for i in ...:
        ts = sv.data_1h[i][0]  # 06:00 UTC
        feats = calc_features_at_ts(ts, ctx)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math


# ---------------------------------------------------------------------
# Константы (можно подстроить под себя)
# ---------------------------------------------------------------------
HOUR_MS: int = 3_600_000
HOURS_IN_YEAR: int = 365 * 24  # для annualize RV к масштабу "процентов годовых"

# Окна для перцентиля DVOL (в днях)
PCTL_LOOKBACK_DAYS: int = 180
PCTL_MIN_SAMPLES: int = 60  # минимум значений, чтобы перцентиль считался "надёжно"

# Окна (в часах)
RV_WINDOWS_H: Tuple[int, ...] = (12, 24)
VOLOFVOL_WINDOWS_H: Tuple[int, ...] = (24, 72)

# Для ATR
ATR_SHORT_H: int = 24
ATR_LONG_H: int = 24 * 7

# Что считать "as-of" ценой на ts_ms:
# Мы считаем, что ts_ms — это старт свечи 06:00, а "текущая цена на 06:00" —
# это close последней полностью закрытой свечи, то есть свечи ts_ms - 1h.
USE_PREV_CLOSE_AS_PRICE: bool = True

MISSING: Optional[float] = None


# ---------------------------------------------------------------------
# Контекст: индексы/доступ к данным
# ---------------------------------------------------------------------
@dataclass
class FeatureContext:
    """
    btc_1h: список/np массив свечей, где минимум поля:
        [0]=timestamp_ms, [1]=open, [2]=high, [3]=low, [4]=close
    dvol_1h: dict ts_ms -> {"open":..., "high":..., "low":..., "close":...}
    """
    btc_1h: Sequence[Sequence[Any]]
    dvol_1h: Dict[int, Dict[str, float]]

    # внутреннее: индекс timestamp_ms -> позиция в btc_1h
    _btc_idx: Dict[int, int] = None  # type: ignore

    def __post_init__(self) -> None:
        idx: Dict[int, int] = {}
        for i, row in enumerate(self.btc_1h):
            try:
                ts = int(row[0])
            except Exception:
                continue
            idx[ts] = i
        self._btc_idx = idx

    def btc_row(self, ts_ms: int) -> Optional[Sequence[Any]]:
        i = self._btc_idx.get(ts_ms)
        if i is None:
            return None
        return self.btc_1h[i]

    def dvol_row(self, ts_ms: int) -> Optional[Dict[str, float]]:
        return self.dvol_1h.get(ts_ms)


# ---------------------------------------------------------------------
# Вспомогательные функции математики (без numpy)
# ---------------------------------------------------------------------
def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return math.fsum(xs) / float(len(xs))


def _std_pop(xs: List[float]) -> Optional[float]:
    """Стандартное отклонение (популяционное). Для фичей этого достаточно."""
    n = len(xs)
    if n < 2:
        return None
    m = _mean(xs)
    if m is None:
        return None
    s2 = math.fsum((x - m) * (x - m) for x in xs) / float(n)
    return math.sqrt(s2)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _utc_dow_hour(ts_ms: int) -> Tuple[int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.weekday(), dt.hour


# ---------------------------------------------------------------------
# Доступ к OHLC
# ---------------------------------------------------------------------
def _row_ohlc(row: Sequence[Any]) -> Tuple[float, float, float, float]:
    # ожидаем: [ts, open, high, low, close, ...]
    o = float(row[1])
    h = float(row[2])
    l = float(row[3])
    c = float(row[4])
    return o, h, l, c


def _get_price_asof(ts_ms: int, ctx: FeatureContext) -> Optional[float]:
    """
    Цена "на ts_ms".
    По умолчанию берём close предыдущей свечи (ts_ms - 1h),
    так как свеча на ts_ms ещё "только открылась".
    """
    if USE_PREV_CLOSE_AS_PRICE:
        r = ctx.btc_row(ts_ms - HOUR_MS)
        if r is None:
            return None
        _, _, _, c = _row_ohlc(r)
        return c

    r = ctx.btc_row(ts_ms)
    if r is None:
        return None
    _, _, _, c = _row_ohlc(r)
    return c


# ---------------------------------------------------------------------
# DVOL фичи
# ---------------------------------------------------------------------
def calc_dvol_percentile(
    ts_ms: int,
    ctx: FeatureContext,
    lookback_days: int = PCTL_LOOKBACK_DAYS,
    min_samples: int = PCTL_MIN_SAMPLES,
    by_same_weekday: bool = False,
) -> Optional[float]:
    """
    Перцентиль DVOL(open) в ts_ms относительно истории последних lookback_days дней.

    Как считаем:
    - собираем DVOL(open) на ts_ms - 24h*k для k=1..lookback_days
    - если by_same_weekday=True, берём только те дни, у которых день недели совпадает
    - считаем долю исторических значений ниже текущего (в процентах)
    """
    cur = ctx.dvol_row(ts_ms)
    if not cur:
        return None
    cur_v = float(cur.get("open", float("nan")))
    if not math.isfinite(cur_v):
        return None

    cur_dow, _ = _utc_dow_hour(ts_ms)

    hist: List[float] = []
    for k in range(1, lookback_days + 1):
        t = ts_ms - k * 24 * HOUR_MS
        r = ctx.dvol_row(t)
        if not r:
            continue
        if by_same_weekday:
            dow_k, _ = _utc_dow_hour(t)
            if dow_k != cur_dow:
                continue
        v = float(r.get("open", float("nan")))
        if math.isfinite(v):
            hist.append(v)

    if len(hist) < min_samples:
        return None

    # чтобы ties не давали скачков, делаем "половину" за равные
    less = 0
    equal = 0
    for x in hist:
        if x < cur_v:
            less += 1
        elif x == cur_v:
            equal += 1

    pct = (float(less) + 0.5 * float(equal)) / float(len(hist)) * 100.0
    return _clamp(pct, 0.0, 100.0)


def calc_dvol_changes(ts_ms: int, ctx: FeatureContext) -> Dict[str, Optional[float]]:
    """
    ΔDVOL и ускорение вокруг ts_ms:
    - dvol_chg_6h  = DVOL(06:00) - DVOL(00:00)
    - dvol_chg_24h = DVOL(06:00) - DVOL(вчера 06:00)
    - dvol_accel_6h = (chg_6h) - (DVOL(00:00)-DVOL(вчера 18:00))
    """
    out: Dict[str, Optional[float]] = {
        "dvol_open": None,
        "dvol_close": None,
        "dvol_chg_6h": None,
        "dvol_chg_24h": None,
        "dvol_accel_6h": None,
    }

    r0 = ctx.dvol_row(ts_ms)
    if not r0:
        return out

    v0 = float(r0.get("open", float("nan")))
    c0 = float(r0.get("close", float("nan")))
    if math.isfinite(v0):
        out["dvol_open"] = v0
    if math.isfinite(c0):
        out["dvol_close"] = c0

    r6 = ctx.dvol_row(ts_ms - 6 * HOUR_MS)
    r24 = ctx.dvol_row(ts_ms - 24 * HOUR_MS)
    r12 = ctx.dvol_row(ts_ms - 12 * HOUR_MS)

    if r6 and math.isfinite(v0):
        v6 = float(r6.get("open", float("nan")))
        if math.isfinite(v6):
            out["dvol_chg_6h"] = v0 - v6

    if r24 and math.isfinite(v0):
        v24 = float(r24.get("open", float("nan")))
        if math.isfinite(v24):
            out["dvol_chg_24h"] = v0 - v24

    # ускорение: сравниваем "последние 6 часов" против "предыдущих 6 часов"
    if r6 and r12 and math.isfinite(v0):
        v6 = float(r6.get("open", float("nan")))
        v12 = float(r12.get("open", float("nan")))
        if math.isfinite(v6) and math.isfinite(v12):
            last_6h = v0 - v6
            prev_6h = v6 - v12
            out["dvol_accel_6h"] = last_6h - prev_6h

    return out


def calc_vol_of_vol(
    ts_ms: int,
    ctx: FeatureContext,
    window_h: int,
) -> Optional[float]:
    """
    Vol-of-vol = насколько DVOL(open) "разбросан" за последние window_h часов до ts_ms.
    Берём значения DVOL(open) на часовых таймстампах: ts_ms-1h, ts_ms-2h, ..., ts_ms-window_h.
    """
    vals: List[float] = []
    for k in range(1, window_h + 1):
        t = ts_ms - k * HOUR_MS
        r = ctx.dvol_row(t)
        if not r:
            continue
        v = float(r.get("open", float("nan")))
        if math.isfinite(v):
            vals.append(v)

    return _std_pop(vals)


# ---------------------------------------------------------------------
# BTC realised volatility (RV) и price-based фичи
# ---------------------------------------------------------------------
def calc_rv_annualised_pct(
    ts_ms: int,
    ctx: FeatureContext,
    window_h: int,
) -> Optional[float]:
    """
    Реализованная волатильность BTC за последние window_h часов до ts_ms,
    приведённая к масштабу "процентов годовых" (сопоставимо с DVOL).

    Как считаем:
    - берём close по часам за окно
    - считаем изменения close между соседними часами
    - оцениваем типичный размер этих изменений (разброс)
    - переводим к годовому масштабу и в проценты
    """
    # нужны closes за window_h+1 точек: от ts_ms-window_h до ts_ms-1h
    closes: List[float] = []
    for k in range(window_h, 0, -1):
        t = ts_ms - k * HOUR_MS
        row = ctx.btc_row(t)
        if row is None:
            return None
        _, _, _, c = _row_ohlc(row)
        closes.append(c)

    # + close последнего закрытого часа (ts_ms-1h)
    last_row = ctx.btc_row(ts_ms - HOUR_MS)
    if last_row is None:
        return None
    _, _, _, c_last = _row_ohlc(last_row)
    closes.append(c_last)

    # returns (log), чтобы корректнее для больших движений
    rets: List[float] = []
    for i in range(1, len(closes)):
        a = closes[i - 1]
        b = closes[i]
        if a <= 0.0 or b <= 0.0:
            return None
        rets.append(math.log(b / a))

    sd = _std_pop(rets)
    if sd is None:
        return None

    # annualize в "процентов годовых"
    ann = sd * math.sqrt(float(HOURS_IN_YEAR))
    return ann * 100.0


def calc_momentum_returns_pct(ts_ms: int, ctx: FeatureContext) -> Dict[str, Optional[float]]:
    """
    Доходность (в процентах) за 3h/6h/24h до ts_ms.
    Берём цену as-of (обычно close предыдущего часа) и сравниваем с ценой N часов назад.
    """
    out: Dict[str, Optional[float]] = {"ret_3h": None, "ret_6h": None, "ret_24h": None}

    p0 = _get_price_asof(ts_ms, ctx)
    if p0 is None or p0 <= 0.0:
        return out

    for h in (3, 6, 24):
        p1 = _get_price_asof(ts_ms - h * HOUR_MS, ctx)
        if p1 is None or p1 <= 0.0:
            continue
        out[f"ret_{h}h"] = (p0 - p1) / p1 * 100.0

    return out


def calc_position_in_range_24h(ts_ms: int, ctx: FeatureContext) -> Dict[str, Optional[float]]:
    """
    position-in-range за последние 24 часа до ts_ms:
    - берём high/low за последние 24 закрытых свечи (ts-24h..ts-1h)
    - берём цену as-of (обычно close ts-1h)
    - считаем где цена лежит внутри диапазона: ближе к low или ближе к high
    """
    out: Dict[str, Optional[float]] = {
        "pos_in_range_24h": None,
        "range_24h_pct": None,
        "hi_24h": None,
        "lo_24h": None,
        "price_asof": None,
    }

    p0 = _get_price_asof(ts_ms, ctx)
    if p0 is None or p0 <= 0.0:
        return out
    out["price_asof"] = p0

    hi = -float("inf")
    lo = float("inf")

    for k in range(24, 0, -1):
        t = ts_ms - k * HOUR_MS
        row = ctx.btc_row(t)
        if row is None:
            return out
        _, h, l, _ = _row_ohlc(row)
        if h > hi:
            hi = h
        if l < lo:
            lo = l

    if not math.isfinite(hi) or not math.isfinite(lo) or hi <= lo:
        return out

    out["hi_24h"] = hi
    out["lo_24h"] = lo

    pos = (p0 - lo) / (hi - lo)
    out["pos_in_range_24h"] = _clamp(pos, 0.0, 1.0)

    rng_pct = (hi - lo) / p0 * 100.0
    out["range_24h_pct"] = rng_pct
    return out


def calc_atr(ts_ms: int, ctx: FeatureContext, window_h: int) -> Optional[float]:
    """
    ATR (в долларах) за последние window_h закрытых свечей до ts_ms.
    True Range считаем так:
    - максимум из: (high-low), |high-prev_close|, |low-prev_close|
    ATR = среднее True Range по окну.
    """
    trs: List[float] = []
    prev_close: Optional[float] = None

    # идём по свечам: ts-window_h .. ts-1h
    for k in range(window_h, 0, -1):
        t = ts_ms - k * HOUR_MS
        row = ctx.btc_row(t)
        if row is None:
            return None
        _, h, l, c = _row_ohlc(row)

        if prev_close is None:
            # для первой свечи prev_close берём close предыдущего часа
            prev_row = ctx.btc_row(t - HOUR_MS)
            if prev_row is None:
                return None
            _, _, _, prev_c = _row_ohlc(prev_row)
            prev_close = prev_c

        tr1 = h - l
        tr2 = abs(h - prev_close)
        tr3 = abs(l - prev_close)
        tr = max(tr1, tr2, tr3)
        trs.append(tr)

        prev_close = c

    m = _mean(trs)
    return m


def calc_atr_features(ts_ms: int, ctx: FeatureContext) -> Dict[str, Optional[float]]:
    """
    ATR 24h, ATR 7d и их отношение (compression proxy).
    """
    out: Dict[str, Optional[float]] = {
        "atr_24h": None,
        "atr_7d": None,
        "atr_ratio_24h_7d": None,
    }

    a24 = calc_atr(ts_ms, ctx, ATR_SHORT_H)
    a7d = calc_atr(ts_ms, ctx, ATR_LONG_H)

    out["atr_24h"] = a24
    out["atr_7d"] = a7d

    if a24 is not None and a7d is not None and a7d > 0.0:
        out["atr_ratio_24h_7d"] = a24 / a7d

    return out


# ---------------------------------------------------------------------
# Главная функция: всё вместе
# ---------------------------------------------------------------------
def calc_features_at_ts(ts_ms: int, ctx: FeatureContext) -> Dict[str, Any]:
    """
    Считает набор фичей на момент ts_ms (у тебя это 06:00 UTC каждого дня).

    Возвращает словарь. Если чего-то не хватает в данных — соответствующее поле будет None.
    """
    dow, hour = _utc_dow_hour(ts_ms)

    out: Dict[str, Any] = {
        "ts_ms": int(ts_ms),
        "dow_utc": dow,         # 0=Mon ... 6=Sun
        "hour_utc": hour,
        "is_weekend": bool(dow >= 5),
    }

    # DVOL уровень + изменения
    out.update(calc_dvol_changes(ts_ms, ctx))

    # DVOL percentile
    out["dvol_percentile_180d"] = calc_dvol_percentile(
        ts_ms, ctx, lookback_days=PCTL_LOOKBACK_DAYS, min_samples=PCTL_MIN_SAMPLES, by_same_weekday=False
    )
    out["dvol_percentile_180d_dow"] = calc_dvol_percentile(
        ts_ms, ctx, lookback_days=PCTL_LOOKBACK_DAYS, min_samples=PCTL_MIN_SAMPLES, by_same_weekday=True
    )

    # Vol-of-vol
    for w in VOLOFVOL_WINDOWS_H:
        out[f"vol_of_vol_{w}h"] = calc_vol_of_vol(ts_ms, ctx, window_h=w)

    # RV и DVOL-RV
    for w in RV_WINDOWS_H:
        rv = calc_rv_annualised_pct(ts_ms, ctx, window_h=w)
        out[f"rv_{w}h_pct"] = rv
        dvol_open = out.get("dvol_open")
        if isinstance(dvol_open, (int, float)) and rv is not None:
            out[f"dvol_minus_rv_{w}h"] = float(dvol_open) - float(rv)
        else:
            out[f"dvol_minus_rv_{w}h"] = None

    # Цена/моментум/диапазон
    out.update(calc_momentum_returns_pct(ts_ms, ctx))
    out.update(calc_position_in_range_24h(ts_ms, ctx))

    # ATR compression proxy
    out.update(calc_atr_features(ts_ms, ctx))

    return out
