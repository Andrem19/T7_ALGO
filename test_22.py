# combo_profit_search.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import itertools
import math
import heapq
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable

import numpy as np
import pandas as pd

import shared_vars as sv
_HEAP_SEQ = itertools.count()

# ======================================================================
# НАСТРОЙКИ (крутите тут)
# ======================================================================

CSV_PATH = "vector.csv"

# Сколько признаков в правиле: 1, 2 или 3 (3 может быть очень тяжело по времени)
MAX_VAR = 1

BASE_F = {}

# Максимальная уникальность столбца, чтобы столбец попадал в перебор
MAX_UNIQUE = 24

# Сколько тайм-бинов для стабильности и просадки (больше = точнее, но тяжелее)
BINS = 24

# Какие столбцы нельзя использовать как признаки
DEFAULT_EXCLUDE_COLS = (
    "tm_ms",
    "feer_and_greed", "fg_stock", "feer_and_greed", "cl_1d", "cl_4h", "cl_1h", "cl_15m", "atr", "rsi", "iv_est", "reg_d", "reg_h", "hill",
    "rsi_1", "sp500", "atr_1", "iv_est_1", "squize_index_1", "vix",
)

# Какие профит-колонки проверяем
DEFAULT_PROFIT_COLS = ("profit_1", "profit_2")

# Период, который берём из CSV (по tm_ms)
sv.START = datetime(2020, 1, 1)
sv.END = datetime(2025, 1, 1)


# -----------------------------
# Ограничения / фильтры
# -----------------------------
DEFAULT_MIN_ROWS = 150

# Не позволять правилу “захватить почти всё”:
# если в каком-то признаке выбрано слишком много значений — правило отбрасывается
DEFAULT_MAX_SELECTED_FRACTION_PER_COL = 0.85

# Минимальные требования к качеству (если не выполнено — правило не рассматриваем)
DEFAULT_REQUIRE_POSITIVE_MEAN = True
DEFAULT_MIN_MEAN = 0.0        # средняя прибыль на сделку должна быть не меньше
DEFAULT_MIN_WINRATE = 0.0     # winrate должен быть не меньше
DEFAULT_MAX_DD_REL = None     # например 1.5, или None чтобы не ограничивать


# -----------------------------
# Целевая функция (веса)
# -----------------------------
# Важно: mean в ваших данных может быть порядка 0.001..0.05 (зависит от нормировки профита).
# mean_scale переводит mean в “удобный масштаб”, чтобы веса имели смысл.
DEFAULT_OBJECTIVE = {
    "mean_scale": 0.01,     # mean делим на это значение, чтобы привести к масштабу около 1
    "w_mean": 1.0,          # хотим высокую среднюю прибыль
    "w_winrate": 0.7,       # хотим высокий winrate
    "w_stability": 0.8,     # хотим стабильность по времени
    "w_drawdown": 1.0,      # хотим маленькую просадку (это штраф)
    "w_rows": 0.1,          # небольшой бонус за размер выборки
    "w_complexity": 0.25,   # небольшой бонус за “узкое” правило (чтобы не брать всё)
}

# Как именно считать “стабильность” (веса внутренних компонентов)
DEFAULT_STABILITY_WEIGHTS = {
    "w_coverage": 0.35,     # доля бинов, где правило реально встречалось
    "w_evenness": 0.45,     # равномерность распределения |прибыль по бинам|
    "w_pos_bins": 0.30,     # доля положительных бинов среди бинов, где есть сделки
    "w_top_share": 0.40,    # штраф за концентрацию прибыли в топ-бинах
    "w_neg_bins": 0.25,     # штраф за долю отрицательных бинов
}

# Как строить тайм-бины:
#  - "rows": равные по числу строк сегменты (быстро и устойчиво)
#  - "time": равные по времени сегменты (нужен tm_ms, полезно если пропуски/неровный шаг)
DEFAULT_TIMELINE = {
    "bins": BINS,
    "top_frac": 0.2,        # доля топ-бинов для оценки концентрации
    "bin_mode": "rows",     # "rows" или "time"
}

# Выводить топ-N кандидатов на каждый набор колонок и каждый profit_col
DEFAULT_TOP_K = 3

# Пересчитать “точную” просадку на уровне сделок (по реальным строкам) для TOP_K лучших
# Это уже не влияет на поиск (по умолчанию), но даёт честную метрику в печати.
DEFAULT_RECALC_EXACT_DD_FOR_TOP = True

# Прогресс-бар
DEFAULT_PROGRESS_UPDATE_EVERY = 2048


# ======================================================================
# ДАННЫЕ РЕЗУЛЬТАТОВ
# ======================================================================

@dataclass(frozen=True)
class RuleCandidate:
    profit_col: str
    cols: Tuple[str, ...]
    bit_masks: Dict[str, int]             # col -> bitmask выбранных значений
    rule_values: Dict[str, Tuple[Any, ...]]
    rule_text: str

    n_rows: int
    profit_sum: float
    profit_mean: float
    winrate: float

    stability_score: float
    timeline_bins: int
    timeline_coverage: float
    timeline_evenness: float
    timeline_top_share: float
    timeline_pos_bins: int
    timeline_neg_bins: int

    dd_abs_bins: float
    dd_rel_bins: float

    # опционально, если пересчитали по реальным сделкам
    dd_abs_exact: Optional[float] = None
    dd_rel_exact: Optional[float] = None

    complexity: float = 0.0              # средняя доля выбранных значений по колонкам (0..1)
    score: float = float("-inf")


class _Progress:
    """
    Единый прогресс-бар с ETA.
    Обновление идёт батчами, чтобы не тормозить на миллионах итераций.
    """
    def __init__(self, total: int, enabled: bool = True, update_every: int = DEFAULT_PROGRESS_UPDATE_EVERY):
        self.total = int(total)
        self.enabled = bool(enabled)
        self.update_every = int(update_every)

        self._acc = 0
        self._done = 0
        self._t0 = time.monotonic()
        self._last_print = self._t0

        self._pbar = None
        if self.enabled:
            try:
                from tqdm import tqdm  # type: ignore
                self._pbar = tqdm(total=self.total, unit="step", dynamic_ncols=True)
            except Exception:
                self._pbar = None

    def add(self, n: int = 1) -> None:
        if self.total <= 0:
            return
        self._acc += int(n)
        if self._acc < self.update_every:
            return
        self._flush()

    def _flush(self) -> None:
        if self._acc <= 0:
            return

        inc = self._acc
        self._acc = 0
        self._done += inc

        if self._pbar is not None:
            self._pbar.update(inc)
            return

        now = time.monotonic()
        if now - self._last_print >= 2.0:
            self._last_print = now
            elapsed = now - self._t0
            done = min(self._done, self.total)
            frac = done / self.total if self.total > 0 else 1.0
            if frac > 0:
                eta = elapsed * (1.0 - frac) / frac
            else:
                eta = float("inf")
            print(f"[progress] {done}/{self.total} ({frac*100:.1f}%) elapsed={elapsed:.1f}s eta~{eta:.1f}s")

    def close(self) -> None:
        self._flush()
        if self._pbar is not None:
            self._pbar.close()


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def _dt_to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)

def _parse_simple_comparator(text: str) -> Tuple[str, str]:
    """
    Парсим строку вида:
      '>0.33', '>= 10', '<=5', '!= 1', '==foo'
    Возвращаем (op, rhs).
    Если оператор не найден — op='' и rhs=text.
    """
    t = str(text).strip()
    for op in (">=", "<=", "!=", "==", ">", "<"):
        if t.startswith(op):
            return op, t[len(op):].strip()
    return "", t


def _apply_base_filter(df: pd.DataFrame, base_filter: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, str]:
    """
    Применяет фиксированное базовое условие к df и возвращает:
      (df_filtered, base_text)

    base_filter формат:
      {"d": [3]}                    -> d in (3,)
      {"hill": [0,2]}               -> hill in (0,2)
      {"iv_est": ">0.33"}           -> iv_est > 0.33
      {"rsi": "<=70"}               -> rsi <= 70

    Для списков/кортежей/множеств используем isin.
    Для строк с операторами: > >= < <= == !=
    """
    if not base_filter:
        return df, ""

    mask = np.ones(len(df), dtype=np.bool_)
    parts: List[str] = []

    for col, cond in base_filter.items():
        if col not in df.columns:
            raise ValueError(f"BASE_F column '{col}' not found in CSV.")

        s = df[col]

        # Строка-компаратор: '>0.33', '<=70', '!=1', ...
        if isinstance(cond, str):
            op, rhs = _parse_simple_comparator(cond)

            if op in (">", ">=", "<", "<="):
                x = pd.to_numeric(s, errors="coerce")
                rhs_f = float(rhs)

                if op == ">":
                    m = (x > rhs_f)
                elif op == ">=":
                    m = (x >= rhs_f)
                elif op == "<":
                    m = (x < rhs_f)
                else:  # "<="
                    m = (x <= rhs_f)

                mask &= m.to_numpy(dtype=np.bool_, copy=False)
                parts.append(f"{col} {op} {rhs_f}")

            elif op in ("==", "!="):
                # Сначала пробуем как число, если не выходит — как строку
                rhs_val: Any
                try:
                    rhs_val = float(rhs)
                    # если колонка целочисленная — можно привести
                    # но это не обязательно, сравнение float/int в pandas нормально работает
                except Exception:
                    rhs_val = rhs

                if op == "==":
                    m = (s == rhs_val)
                else:
                    m = (s != rhs_val)

                mask &= m.to_numpy(dtype=np.bool_, copy=False)
                parts.append(f"{col} {op} {rhs_val}")

            else:
                # Если строка без оператора — трактуем как равенство
                m = (s.astype(str) == str(cond))
                mask &= m.to_numpy(dtype=np.bool_, copy=False)
                parts.append(f"{col} == {cond}")

            continue

        # Список значений: {"d":[3,4]} -> isin
        if isinstance(cond, (list, tuple, set, np.ndarray, pd.Series)):
            allowed = list(cond)
            m = s.isin(allowed)
            mask &= m.to_numpy(dtype=np.bool_, copy=False)
            parts.append(f"{col} in {tuple(allowed)}")
            continue

        # Скаляр: {"d":3}
        m = (s == cond)
        mask &= m.to_numpy(dtype=np.bool_, copy=False)
        parts.append(f"{col} == {cond}")

    base_txt = " and ".join(parts)
    df2 = df[mask].reset_index(drop=True)

    return df2, base_txt

def _safe_factorize_series(s: pd.Series) -> Tuple[np.ndarray, List[Any]]:
    """
    Факторизация без проблем с NaN.
    Для чисел: NaN заменяем на (минимум - 1).
    Для не-чисел: NaN заменяем на строковый маркер.
    """
    if pd.api.types.is_numeric_dtype(s):
        if s.isna().any():
            mn = pd.to_numeric(s, errors="coerce").min(skipna=True)
            sentinel = (float(mn) - 1.0) if mn is not None and not pd.isna(mn) else -1.0
            s2 = pd.to_numeric(s, errors="coerce").fillna(sentinel)
        else:
            s2 = pd.to_numeric(s, errors="coerce")
        codes, uniques = pd.factorize(s2, sort=True)
        return codes.astype(np.int64, copy=False), uniques.to_list()
    else:
        s2 = s.astype(object).where(~s.isna(), "__NA__")
        codes, uniques = pd.factorize(s2, sort=True)
        return codes.astype(np.int64, copy=False), uniques.to_list()


def _factorize_columns(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in cols:
        codes, uniques = _safe_factorize_series(df[c])
        out[c] = {"codes": codes, "uniques": uniques, "k": int(len(uniques))}
    return out


def _popcount(x: int) -> int:
    # Python 3.8+ умеет int.bit_count()
    return int(x.bit_count())


def _mask_to_values(mask: int, uniques: Sequence[Any]) -> Tuple[Any, ...]:
    """
    Быстро достаём значения по установленным битам маски.
    Идём только по включённым битам (а не по всем uniques).
    """
    vals: List[Any] = []
    m = int(mask)
    while m:
        lsb = m & -m
        bit = int(lsb.bit_length() - 1)
        vals.append(uniques[bit])
        m ^= lsb
    return tuple(vals)



def _rule_text(rule_values: Dict[str, Tuple[Any, ...]]) -> str:
    parts = []
    for col, vals in rule_values.items():
        parts.append(f"{col} in {vals}")
    return " and ".join(parts)


def _make_bin_id(df: pd.DataFrame, *, bins: int, bin_mode: str) -> np.ndarray:
    n = int(len(df))
    if bins <= 0:
        raise ValueError("bins must be > 0")

    mode = str(bin_mode).strip().lower()
    if mode == "rows":
        idx = np.arange(n, dtype=np.int64)
        b = (idx * int(bins) // max(1, n)).astype(np.int64, copy=False)
        return np.minimum(b, int(bins) - 1)

    if mode == "time":
        if "tm_ms" not in df.columns:
            raise ValueError("bin_mode='time' requires 'tm_ms' column")
        tm = pd.to_numeric(df["tm_ms"], errors="coerce").to_numpy(dtype=np.int64, copy=False)
        t0 = int(np.min(tm))
        t1 = int(np.max(tm))
        span = max(1, t1 - t0 + 1)
        b = ((tm - t0) * int(bins) // span).astype(np.int64, copy=False)
        return np.minimum(np.maximum(b, 0), int(bins) - 1)

    raise ValueError("bin_mode must be 'rows' or 'time'")


def _bincount_tensor(
    flat_idx: np.ndarray,
    size: int,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weights is None:
        out = np.bincount(flat_idx, minlength=size).astype(np.int64, copy=False)
    else:
        out = np.bincount(flat_idx, weights=weights, minlength=size).astype(np.float64, copy=False)
    return out


def _bincount_bin_tensor(
    bin_id: np.ndarray,
    flat_idx: np.ndarray,
    n_bins: int,
    size: int,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    # joint index: bin * size + flat_idx
    joint = (bin_id.astype(np.int64, copy=False) * int(size) + flat_idx.astype(np.int64, copy=False)).astype(np.int64, copy=False)
    full = int(n_bins) * int(size)
    if weights is None:
        out = np.bincount(joint, minlength=full).astype(np.int64, copy=False)
    else:
        out = np.bincount(joint, weights=weights, minlength=full).astype(np.float64, copy=False)
    return out.reshape(int(n_bins), int(size))


def _calc_evenness_and_topshare(abs_bin_profit: np.ndarray, top_frac: float) -> Tuple[float, float]:
    total_abs = float(abs_bin_profit.sum())
    if total_abs <= 0.0:
        return 0.0, 0.0

    n_bins = int(abs_bin_profit.shape[0])
    top_k = int(max(1, math.ceil(n_bins * float(top_frac))))
    top_share = float(np.sort(abs_bin_profit)[::-1][:top_k].sum() / total_abs)

    nonzero = abs_bin_profit[abs_bin_profit > 0]
    if nonzero.size <= 1:
        evenness = 1.0
    else:
        probs = nonzero / nonzero.sum()
        ent = float(-(probs * np.log(probs)).sum())
        ent_max = float(np.log(nonzero.size))
        evenness = float(ent / ent_max) if ent_max > 0 else 0.0

    return evenness, top_share


def _calc_drawdown_from_bins(bin_profit: np.ndarray) -> float:
    """
    Просадка по бинам:
    идём по бинам, накапливаем кумулятив, держим пик и считаем худшее падение от пика.
    """
    peak = 0.0
    cur = 0.0
    max_dd = 0.0

    arr = np.asarray(bin_profit, dtype=np.float64).ravel()
    for x in arr:
        cur += float(x)
        if cur > peak:
            peak = cur
        dd = peak - cur
        if dd > max_dd:
            max_dd = dd

    return float(max_dd)



def _calc_stability(
    *,
    bin_profit: np.ndarray,
    bin_count: np.ndarray,
    top_frac: float,
    stab_w: Dict[str, float],
) -> Tuple[float, float, float, float, int, int]:
    n_bins = int(bin_profit.shape[0])
    has = bin_count > 0

    bins_hit = int(has.sum())
    coverage = float(bins_hit / n_bins) if n_bins > 0 else 0.0

    pos_bins = int(((bin_profit > 0) & has).sum())
    neg_bins = int(((bin_profit < 0) & has).sum())

    denom = max(1, bins_hit)
    pos_frac = float(pos_bins / denom)
    neg_frac = float(neg_bins / denom)

    abs_profit = np.abs(bin_profit)
    evenness, top_share = _calc_evenness_and_topshare(abs_profit, top_frac=float(top_frac))

    # stability_score: чем больше, тем лучше
    # - хотим много покрытых бинов
    # - хотим равномерность
    # - хотим, чтобы больше бинов было плюсовыми
    # - штрафуем концентрацию в топ-бинах
    # - штрафуем долю минусовых бинов
    score = (
        float(stab_w.get("w_coverage", 0.0)) * coverage
        + float(stab_w.get("w_evenness", 0.0)) * evenness
        + float(stab_w.get("w_pos_bins", 0.0)) * pos_frac
        - float(stab_w.get("w_top_share", 0.0)) * top_share
        - float(stab_w.get("w_neg_bins", 0.0)) * neg_frac
    )

    # мягкая защита от отрицательных значений
    score = float(max(-1.0, min(1.5, score)))

    return score, coverage, evenness, top_share, pos_bins, neg_bins

STAB_SCORE_MIN = -1.0
STAB_SCORE_MAX = 1.5


def _score_upper_bound(
    *,
    profit_mean: float,
    winrate: float,
    n_rows: int,
    n_total_rows: int,
    complexity: float,
    obj: Dict[str, float],
) -> float:
    """
    Строго безопасная верхняя граница score для кандидата.

    Идея: берем реальный вклад mean/winrate/rows/complexity,
    а для стабильности и просадки подставляем наилучшие теоретически возможные значения.

    Это дает ОПТИМИСТИЧНЫЙ score. Если даже он <= текущего порога TOP-K,
    кандидата можно пропускать без потери правильности.
    """
    mean_scale = float(obj.get("mean_scale", 1.0))
    mean_norm = float(profit_mean / mean_scale) if mean_scale != 0 else float(profit_mean)

    # rows_bonus в диапазоне 0..1
    if n_total_rows > 0:
        rows_bonus = float(math.log1p(n_rows) / math.log1p(n_total_rows))
    else:
        rows_bonus = 0.0

    # complexity_bonus в диапазоне 0..1 (меньше complexity -> больше бонус)
    c = float(max(0.0, min(1.0, complexity)))
    complexity_bonus = float(1.0 - c)

    w_mean = float(obj.get("w_mean", 0.0))
    w_win = float(obj.get("w_winrate", 0.0))
    w_stab = float(obj.get("w_stability", 0.0))
    w_dd = float(obj.get("w_drawdown", 0.0))
    w_rows = float(obj.get("w_rows", 0.0))
    w_cmp = float(obj.get("w_complexity", 0.0))

    # Для stability берём лучший возможный край, зависящий от знака веса.
    stab_best = STAB_SCORE_MAX if w_stab >= 0.0 else STAB_SCORE_MIN

    # Для drawdown term:
    # score содержит "- w_drawdown * dd_rel".
    # Если w_drawdown >= 0, то лучший случай dd_rel = 0.
    # Если w_drawdown < 0, то dd_rel "улучшает" score и верхняя граница становится неограниченной.
    # В таком случае prune по dd делать нельзя -> просто не используем dd в bound (делаем bound очень большим).
    if w_dd >= 0.0:
        dd_best_term = 0.0  # dd_rel=0
        dd_part = -w_dd * dd_best_term
    else:
        # Нельзя безопасно ограничить сверху вклад dd -> отключаем prune по bound (возвращаем +inf).
        return float("inf")

    ub = (
        w_mean * mean_norm
        + w_win * float(winrate)
        + w_stab * float(stab_best)
        + dd_part
        + w_rows * rows_bonus
        + w_cmp * complexity_bonus
    )
    return float(ub)

def _score_candidate(
    *,
    profit_mean: float,
    winrate: float,
    stability_score: float,
    dd_rel: float,
    n_rows: int,
    n_total_rows: int,
    complexity: float,
    obj: Dict[str, float],
    # новое (необязательное): ускорение log1p
    log1p_table: Optional[np.ndarray] = None,
    log1p_total: Optional[float] = None,
) -> float:
    mean_scale = float(obj.get("mean_scale", 1.0))
    mean_norm = float(profit_mean / mean_scale) if mean_scale != 0 else float(profit_mean)

    # rows_bonus: логарифм от (один плюс количество строк) делим на логарифм от (один плюс весь объём)
    if n_total_rows > 0:
        if (
            log1p_table is not None
            and log1p_total is not None
            and 0 <= int(n_rows) < int(len(log1p_table))
            and float(log1p_total) > 0.0
        ):
            rows_bonus = float(log1p_table[int(n_rows)]) / float(log1p_total)
        else:
            rows_bonus = float(math.log1p(int(n_rows)) / math.log1p(int(n_total_rows)))
    else:
        rows_bonus = 0.0

    # Complexity: это доля выбранных значений (0..1). Меньше — лучше.
    c = float(max(0.0, min(1.0, float(complexity))))
    complexity_bonus = float(1.0 - c)

    score = (
        float(obj.get("w_mean", 0.0)) * mean_norm
        + float(obj.get("w_winrate", 0.0)) * float(winrate)
        + float(obj.get("w_stability", 0.0)) * float(stability_score)
        - float(obj.get("w_drawdown", 0.0)) * float(dd_rel)
        + float(obj.get("w_rows", 0.0)) * float(rows_bonus)
        + float(obj.get("w_complexity", 0.0)) * float(complexity_bonus)
    )
    return float(score)



def _calc_exact_drawdown_on_rows(profit: np.ndarray, include_mask: np.ndarray) -> float:
    """
    Точная просадка на уровне сделок (строк).
    Идём по выбранным сделкам в хронологическом порядке и считаем худшее падение от локального пика.
    """
    peak = 0.0
    cur = 0.0
    max_dd = 0.0

    p = profit[include_mask]
    for x in p:
        cur += float(x)
        if cur > peak:
            peak = cur
        dd = peak - cur
        if dd > max_dd:
            max_dd = dd

    return float(max_dd)



# ======================================================================
# ПРЕДВЫЧИСЛЕНИЯ ДЛЯ ПЕРЕБОРА МАСОК
# ======================================================================
# ----------------------------------------------------------------------
# FAST stability + drawdown in one pass (optionally with Numba)
# ----------------------------------------------------------------------

try:
    from numba import njit  # type: ignore
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _top_k_bins_count(bins: int, top_frac: float) -> int:
    return int(max(1, math.ceil(int(bins) * float(top_frac))))


def _clamp_stab(score: float) -> float:
    if score < -1.0:
        return -1.0
    if score > 1.5:
        return 1.5
    return float(score)


def _stability_dd_py(
    bin_profit: np.ndarray,
    bin_count: np.ndarray,
    top_k_bins: int,
    w_cov: float,
    w_even: float,
    w_pos: float,
    w_top: float,
    w_neg: float,
) -> Tuple[float, float, float, float, int, int, float]:
    bp = np.asarray(bin_profit, dtype=np.float64).ravel()
    bc = np.asarray(bin_count, dtype=np.int64).ravel()

    n_bins = int(bp.shape[0])

    # coverage / pos_bins / neg_bins
    bins_hit = 0
    pos_bins = 0
    neg_bins = 0

    # drawdown
    peak = 0.0
    cur = 0.0
    max_dd = 0.0

    # abs-profit aggregates for evenness/top_share
    abs_vals = np.empty(n_bins, dtype=np.float64)
    total_abs = 0.0
    nz_cnt = 0

    for i in range(n_bins):
        c = int(bc[i])
        x = float(bp[i])

        if c > 0:
            bins_hit += 1
            if x > 0.0:
                pos_bins += 1
            elif x < 0.0:
                neg_bins += 1

        cur += x
        if cur > peak:
            peak = cur
        dd = peak - cur
        if dd > max_dd:
            max_dd = dd

        ax = x if x >= 0.0 else -x
        abs_vals[i] = ax
        total_abs += ax
        if ax > 0.0:
            nz_cnt += 1

    coverage = float(bins_hit / n_bins) if n_bins > 0 else 0.0
    denom = bins_hit if bins_hit > 0 else 1
    pos_frac = float(pos_bins / denom)
    neg_frac = float(neg_bins / denom)

    # top_share
    if total_abs <= 0.0:
        top_share = 0.0
        evenness = 0.0
    else:
        # top_share через partition (быстро)
        k = int(top_k_bins)
        if k >= n_bins:
            top_sum = float(total_abs)
        else:
            part = np.partition(abs_vals, n_bins - k)
            top_sum = float(part[n_bins - k :].sum())
        top_share = float(top_sum / total_abs)

        # evenness = энтропия по ненулевым |profit|
        if nz_cnt <= 1:
            evenness = 1.0
        else:
            ent = 0.0
            for i in range(n_bins):
                ax = float(abs_vals[i])
                if ax > 0.0:
                    p = ax / total_abs
                    ent -= p * math.log(p)
            ent_max = math.log(float(nz_cnt))
            evenness = float(ent / ent_max) if ent_max > 0.0 else 0.0

    stab = (
        float(w_cov) * float(coverage)
        + float(w_even) * float(evenness)
        + float(w_pos) * float(pos_frac)
        - float(w_top) * float(top_share)
        - float(w_neg) * float(neg_frac)
    )
    stab = _clamp_stab(stab)

    return float(stab), float(coverage), float(evenness), float(top_share), int(pos_bins), int(neg_bins), float(max_dd)


if _HAVE_NUMBA:
    @njit(cache=True)
    def _stability_dd_numba(
        bp: np.ndarray,
        bc: np.ndarray,
        top_k_bins: int,
        w_cov: float,
        w_even: float,
        w_pos: float,
        w_top: float,
        w_neg: float,
    ) -> Tuple[float, float, float, float, int, int, float]:
        n_bins = bp.shape[0]

        bins_hit = 0
        pos_bins = 0
        neg_bins = 0

        peak = 0.0
        cur = 0.0
        max_dd = 0.0

        total_abs = 0.0
        nz_cnt = 0

        abs_vals = np.empty(n_bins, dtype=np.float64)

        for i in range(n_bins):
            c = int(bc[i])
            x = float(bp[i])

            if c > 0:
                bins_hit += 1
                if x > 0.0:
                    pos_bins += 1
                elif x < 0.0:
                    neg_bins += 1

            cur += x
            if cur > peak:
                peak = cur
            dd = peak - cur
            if dd > max_dd:
                max_dd = dd

            ax = x if x >= 0.0 else -x
            abs_vals[i] = ax
            total_abs += ax
            if ax > 0.0:
                nz_cnt += 1

        coverage = bins_hit / n_bins if n_bins > 0 else 0.0
        denom = bins_hit if bins_hit > 0 else 1
        pos_frac = pos_bins / denom
        neg_frac = neg_bins / denom

        if total_abs <= 0.0:
            top_share = 0.0
            evenness = 0.0
        else:
            # top_share: сумма топ-k значений (через простой top-k подбор)
            k = top_k_bins
            if k >= n_bins:
                top_sum = total_abs
            else:
                # держим массив top значений и минимум внутри него
                top = np.zeros(k, dtype=np.float64)
                filled = 0
                min_idx = 0
                min_val = 0.0

                for i in range(n_bins):
                    v = abs_vals[i]
                    if filled < k:
                        top[filled] = v
                        filled += 1
                        if filled == k:
                            # найдём минимум
                            min_idx = 0
                            min_val = top[0]
                            for j in range(1, k):
                                if top[j] < min_val:
                                    min_val = top[j]
                                    min_idx = j
                    else:
                        if v > min_val:
                            top[min_idx] = v
                            # пересчёт минимума
                            min_idx = 0
                            min_val = top[0]
                            for j in range(1, k):
                                if top[j] < min_val:
                                    min_val = top[j]
                                    min_idx = j

                top_sum = 0.0
                for j in range(k):
                    top_sum += top[j]

            top_share = top_sum / total_abs

            if nz_cnt <= 1:
                evenness = 1.0
            else:
                ent = 0.0
                for i in range(n_bins):
                    v = abs_vals[i]
                    if v > 0.0:
                        p = v / total_abs
                        ent -= p * math.log(p)
                ent_max = math.log(float(nz_cnt))
                evenness = ent / ent_max if ent_max > 0.0 else 0.0

        stab = (
            w_cov * coverage
            + w_even * evenness
            + w_pos * pos_frac
            - w_top * top_share
            - w_neg * neg_frac
        )
        if stab < -1.0:
            stab = -1.0
        elif stab > 1.5:
            stab = 1.5

        return stab, coverage, evenness, top_share, pos_bins, neg_bins, max_dd


def _calc_stability_and_dd_fast(
    *,
    bin_profit: np.ndarray,
    bin_count: np.ndarray,
    top_k_bins: int,
    stab_w: Dict[str, float],
) -> Tuple[float, float, float, float, int, int, float]:
    w_cov = float(stab_w.get("w_coverage", 0.0))
    w_even = float(stab_w.get("w_evenness", 0.0))
    w_pos = float(stab_w.get("w_pos_bins", 0.0))
    w_top = float(stab_w.get("w_top_share", 0.0))
    w_neg = float(stab_w.get("w_neg_bins", 0.0))

    bp = np.asarray(bin_profit, dtype=np.float64).ravel()
    bc = np.asarray(bin_count, dtype=np.int64).ravel()

    if _HAVE_NUMBA:
        return _stability_dd_numba(bp, bc, int(top_k_bins), w_cov, w_even, w_pos, w_top, w_neg)  # type: ignore
    else:
        return _stability_dd_py(bp, bc, int(top_k_bins), w_cov, w_even, w_pos, w_top, w_neg)

def _build_log1p_table(n: int) -> Tuple[np.ndarray, float]:
    n = int(n)
    tbl = np.empty(n + 1, dtype=np.float64)
    for i in range(n + 1):
        tbl[i] = math.log1p(i)
    total = float(tbl[n]) if n > 0 else 1.0
    return tbl, total

def _subset_sums_1d(vec: np.ndarray) -> np.ndarray:
    """
    Для vec длины k возвращает массив sums длины (1<<k),
    где sums[mask] = сумма vec по выбранным битам.
    """
    vec = np.asarray(vec)
    k = int(vec.shape[0])
    out = np.zeros((1 << k,), dtype=np.float64)
    for mask in range(1, (1 << k)):
        lsb = mask & -mask
        bit = int(lsb.bit_length() - 1)
        prev = mask ^ lsb
        out[mask] = out[prev] + float(vec[bit])
    return out


def _subset_sums_1d_int(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec)
    k = int(vec.shape[0])
    out = np.zeros((1 << k,), dtype=np.int64)
    for mask in range(1, (1 << k)):
        lsb = mask & -mask
        bit = int(lsb.bit_length() - 1)
        prev = mask ^ lsb
        out[mask] = out[prev] + int(vec[bit])
    return out


def _mask_indices_list(k: int) -> List[np.ndarray]:
    """
    Для каждого mask (0..(1<<k)-1) хранит массив индексов выбранных битов.
    """
    out: List[np.ndarray] = []
    for mask in range(1 << k):
        idx = [i for i in range(k) if ((mask >> i) & 1)]
        out.append(np.asarray(idx, dtype=np.int64))
    return out


# ======================================================================
# ПОИСК ЛУЧШИХ ПРАВИЛ ДЛЯ 1D / 2D / 3D
# ======================================================================

def _push_topk(heap: List[Tuple[float, int, RuleCandidate]], cand: RuleCandidate, top_k: int) -> None:
    """
    heap хранит тройки (score, seq, cand).
    seq нужен как tiebreaker, чтобы при равных score heapq не пытался сравнивать RuleCandidate.
    """
    if top_k <= 0:
        return

    entry = (float(cand.score), int(next(_HEAP_SEQ)), cand)

    if len(heap) < top_k:
        heapq.heappush(heap, entry)
        return

    # сравниваем только по score (heap[0] — худший по score в куче)
    if entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)



def _search_1d(
    *,
    col: str,
    uniques: List[Any],
    k: int,
    profit_col: str,
    sum_by_val: np.ndarray,
    cnt_by_val: np.ndarray,
    pos_by_val: np.ndarray,
    sum_bin_by_val: np.ndarray,
    cnt_bin_by_val: np.ndarray,
    bins: int,
    top_frac: float,
    min_rows: int,
    max_sel_frac: float,
    require_pos_mean: bool,
    min_mean: float,
    min_winrate: float,
    max_dd_rel: Optional[float],
    obj: Dict[str, float],
    stab_w: Dict[str, float],
    n_total_rows: int,
    top_k: int,
    progress: Optional[_Progress],
    log1p_table: np.ndarray,
    log1p_total: float,
) -> List[RuleCandidate]:
    sum_by_val = np.asarray(sum_by_val, dtype=np.float64)
    cnt_by_val = np.asarray(cnt_by_val, dtype=np.int64)
    pos_by_val = np.asarray(pos_by_val, dtype=np.int64)
    sum_bin_by_val = np.asarray(sum_bin_by_val, dtype=np.float64)
    cnt_bin_by_val = np.asarray(cnt_bin_by_val, dtype=np.int64)

    mean_scale = float(obj.get("mean_scale", 1.0))
    inv_mean_scale = (1.0 / mean_scale) if mean_scale != 0.0 else 1.0

    w_mean = float(obj.get("w_mean", 0.0))
    w_win = float(obj.get("w_winrate", 0.0))
    w_stab = float(obj.get("w_stability", 0.0))
    w_dd = float(obj.get("w_drawdown", 0.0))
    w_rows = float(obj.get("w_rows", 0.0))
    w_cplx = float(obj.get("w_complexity", 0.0))

    # для безопасного верхнего бонда
    stab_best = 1.5 if w_stab >= 0.0 else -1.0
    can_bound = (w_dd >= 0.0)

    inv_k = 1.0 / float(max(1, int(k)))
    max_sel_thr = float(max_sel_frac) * float(k)

    top_k_bins = int(max(1, math.ceil(float(bins) * float(top_frac))))

    heap: List[Tuple[float, int, RuleCandidate]] = []

    prev_mask = 0
    bits_on = 0

    n_run = 0
    s_run = 0.0
    pos_run = 0

    bin_profit_run = np.zeros(int(bins), dtype=np.float64)
    bin_count_run = np.zeros(int(bins), dtype=np.int64)

    # батч прогресса
    acc = 0
    upd_every = int(progress.update_every) if progress is not None else 0

    full = 1 << int(k)
    for i in range(1, full):
        if progress is not None:
            acc += 1
            if acc >= upd_every:
                progress.add(acc)
                acc = 0

        mask = i ^ (i >> 1)
        delta = mask ^ prev_mask
        bit = int(delta.bit_length() - 1)

        turned_on = ((mask >> bit) & 1) == 1
        if turned_on:
            bits_on += 1
            n_run += int(cnt_by_val[bit])
            s_run += float(sum_by_val[bit])
            pos_run += int(pos_by_val[bit])
            bin_profit_run += sum_bin_by_val[:, bit]
            bin_count_run += cnt_bin_by_val[:, bit]
        else:
            bits_on -= 1
            n_run -= int(cnt_by_val[bit])
            s_run -= float(sum_by_val[bit])
            pos_run -= int(pos_by_val[bit])
            bin_profit_run -= sum_bin_by_val[:, bit]
            bin_count_run -= cnt_bin_by_val[:, bit]

        prev_mask = mask

        if n_run < int(min_rows):
            continue
        if float(bits_on) > max_sel_thr:
            continue

        mean = float(s_run / n_run) if n_run > 0 else float("nan")
        if require_pos_mean and not (mean > 0.0):
            continue
        if mean < float(min_mean):
            continue

        winrate = float(pos_run / n_run) if n_run > 0 else 0.0
        if winrate < float(min_winrate):
            continue

        sel_frac = float(bits_on) * inv_k
        rows_bonus = float(log1p_table[int(n_run)] / log1p_total) if log1p_total > 0.0 else 0.0
        cplx_bonus = float(1.0 - sel_frac)

        mean_norm = float(mean) * inv_mean_scale
        base_score = (
            w_mean * mean_norm
            + w_win * float(winrate)
            + w_rows * rows_bonus
            + w_cplx * cplx_bonus
        )

        # safe pruning: если даже в лучшем случае не обгоняет худшего в heap — пропускаем
        if can_bound and len(heap) >= int(top_k):
            worst = float(heap[0][0])
            ub = base_score + (w_stab * stab_best)  # drawdown в лучшем случае даёт 0 штрафа
            if ub <= worst:
                continue

        stab_score, cov, eve, top_share, pos_bins, neg_bins, dd_abs = _calc_stability_and_dd_fast(
            bin_profit=bin_profit_run,
            bin_count=bin_count_run,
            top_k_bins=top_k_bins,
            stab_w=stab_w,
        )

        dd_scale = max(1e-12, abs(float(s_run)))
        dd_rel = float(dd_abs / dd_scale)

        if max_dd_rel is not None and dd_rel > float(max_dd_rel):
            continue

        score = (
            base_score
            + w_stab * float(stab_score)
            - w_dd * float(dd_rel)
        )

        # не строим объект/строку, если точно не улучшаем top-k
        if len(heap) >= int(top_k) and score <= float(heap[0][0]):
            continue

        vals = _mask_to_values(mask, uniques)
        rule_vals = {col: vals}
        rule_txt = _rule_text(rule_vals)

        cand = RuleCandidate(
            profit_col=profit_col,
            cols=(col,),
            bit_masks={col: int(mask)},
            rule_values=rule_vals,
            rule_text=rule_txt,
            n_rows=int(n_run),
            profit_sum=float(s_run),
            profit_mean=float(mean),
            winrate=float(winrate),
            stability_score=float(stab_score),
            timeline_bins=int(bins),
            timeline_coverage=float(cov),
            timeline_evenness=float(eve),
            timeline_top_share=float(top_share),
            timeline_pos_bins=int(pos_bins),
            timeline_neg_bins=int(neg_bins),
            dd_abs_bins=float(dd_abs),
            dd_rel_bins=float(dd_rel),
            complexity=float(sel_frac),
            score=float(score),
        )
        _push_topk(heap, cand, top_k=int(top_k))

    if progress is not None and acc > 0:
        progress.add(acc)

    out = [e[2] for e in sorted(heap, key=lambda x: x[0], reverse=False)]
    return out




def _search_2d(
    *,
    cols: Tuple[str, str],
    uniques0: List[Any],
    uniques1: List[Any],
    k0: int,
    k1: int,
    profit_col: str,
    sum_2d: np.ndarray,              # (k0,k1)
    cnt_2d: np.ndarray,              # (k0,k1)
    pos_2d: np.ndarray,              # (k0,k1)
    sum_bin_2d: np.ndarray,          # (bins,k0,k1)
    cnt_bin_2d: np.ndarray,          # (bins,k0,k1)
    bins: int,
    top_frac: float,
    min_rows: int,
    max_sel_frac: float,
    require_pos_mean: bool,
    min_mean: float,
    min_winrate: float,
    max_dd_rel: Optional[float],
    obj: Dict[str, float],
    stab_w: Dict[str, float],
    n_total_rows: int,
    top_k: int,
    progress: Optional[_Progress],
    log1p_table: Optional[np.ndarray] = None,
    **_kw: Any,
) -> List[RuleCandidate]:
    orig_c0, orig_c1 = cols
    c0, c1 = orig_c0, orig_c1

    sum_2d = np.asarray(sum_2d, dtype=np.float64)
    cnt_2d = np.asarray(cnt_2d, dtype=np.int64)
    pos_2d = np.asarray(pos_2d, dtype=np.int64)
    sum_bin_2d = np.asarray(sum_bin_2d, dtype=np.float64)
    cnt_bin_2d = np.asarray(cnt_bin_2d, dtype=np.int64)

    # swap чтобы внешняя ось была меньше (меньше перезапусков внутреннего цикла)
    swapped = False
    k0 = int(k0)
    k1 = int(k1)
    if k0 > k1:
        swapped = True
        k0, k1 = k1, k0
        uniques0, uniques1 = uniques1, uniques0
        c0, c1 = c1, c0

        sum_2d = sum_2d.T
        cnt_2d = cnt_2d.T
        pos_2d = pos_2d.T
        sum_bin_2d = np.swapaxes(sum_bin_2d, 1, 2)  # (bins,k0,k1)
        cnt_bin_2d = np.swapaxes(cnt_bin_2d, 1, 2)

    # log1p_total
    log1p_total: float = 1.0
    if log1p_table is not None and 0 <= int(n_total_rows) < int(len(log1p_table)):
        log1p_total = float(log1p_table[int(n_total_rows)]) if n_total_rows > 0 else 1.0

    mean_scale = float(obj.get("mean_scale", 1.0))
    inv_mean_scale = (1.0 / mean_scale) if mean_scale != 0.0 else 1.0

    w_mean = float(obj.get("w_mean", 0.0))
    w_win = float(obj.get("w_winrate", 0.0))
    w_stab = float(obj.get("w_stability", 0.0))
    w_dd = float(obj.get("w_drawdown", 0.0))
    w_rows = float(obj.get("w_rows", 0.0))
    w_cplx = float(obj.get("w_complexity", 0.0))

    stab_best = 1.5 if w_stab >= 0.0 else -1.0
    can_bound = (w_dd >= 0.0)

    inv_k0 = 1.0 / float(max(1, k0))
    inv_k1 = 1.0 / float(max(1, k1))
    max_sel_thr0 = float(max_sel_frac) * float(k0)
    max_sel_thr1 = float(max_sel_frac) * float(k1)

    top_k_bins = int(max(1, math.ceil(float(bins) * float(top_frac))))

    heap: List[Tuple[float, int, RuleCandidate]] = []

    # ---- быстрый прогресс батчами ----
    acc = 0
    upd_every = int(progress.update_every) if progress is not None else 0

    def _padd(n: int) -> None:
        nonlocal acc
        if progress is None:
            return
        acc += int(n)
        if acc >= upd_every:
            progress.add(acc)
            acc = 0

    # ---- заполнение heap начальными “синглтонами” (ускоряет prune дальше) ----
    for i0 in range(k0):
        for i1 in range(k1):
            _padd(1)

            n_run = int(cnt_2d[i0, i1])
            if n_run < int(min_rows):
                continue

            s_run = float(sum_2d[i0, i1])
            mean = float(s_run / n_run) if n_run > 0 else float("nan")
            if require_pos_mean and not (mean > 0.0):
                continue
            if mean < float(min_mean):
                continue

            pos_run = int(pos_2d[i0, i1])
            winrate = float(pos_run / n_run) if n_run > 0 else 0.0
            if winrate < float(min_winrate):
                continue

            sel0_frac = float(1.0 * inv_k0)
            sel1_frac = float(1.0 * inv_k1)
            complexity = float(0.5 * (sel0_frac + sel1_frac))

            if log1p_table is not None and 0 <= n_run < int(len(log1p_table)) and log1p_total > 0.0:
                rows_bonus = float(log1p_table[n_run]) / float(log1p_total)
            else:
                rows_bonus = float(math.log1p(n_run) / math.log1p(max(1, int(n_total_rows))))

            cplx_bonus = float(1.0 - complexity)
            mean_norm = float(mean) * inv_mean_scale

            base_score = (
                w_mean * mean_norm
                + w_win * float(winrate)
                + w_rows * float(rows_bonus)
                + w_cplx * float(cplx_bonus)
            )

            if can_bound and len(heap) >= int(top_k):
                worst = float(heap[0][0])
                ub = base_score + (w_stab * stab_best)
                if ub <= worst:
                    continue

            bp = sum_bin_2d[:, i0, i1]
            bc = cnt_bin_2d[:, i0, i1]
            stab_score, cov, eve, top_share, pos_bins, neg_bins, dd_abs = _calc_stability_and_dd_fast(
                bin_profit=bp,
                bin_count=bc,
                top_k_bins=top_k_bins,
                stab_w=stab_w,
            )
            dd_scale = max(1e-12, abs(float(s_run)))
            dd_rel = float(dd_abs / dd_scale)
            if max_dd_rel is not None and dd_rel > float(max_dd_rel):
                continue

            score = base_score + w_stab * float(stab_score) - w_dd * float(dd_rel)
            if len(heap) >= int(top_k) and score <= float(heap[0][0]):
                continue

            mask0 = 1 << i0
            mask1 = 1 << i1

            if not swapped:
                vals0 = (uniques0[i0],)
                vals1 = (uniques1[i1],)
                rule_vals = {c0: vals0, c1: vals1}
                bit_masks = {c0: int(mask0), c1: int(mask1)}
                cols_out = (c0, c1)
            else:
                # вернёмся к исходному порядку колонок
                vals_orig0 = (uniques1[i1],)  # orig_c0
                vals_orig1 = (uniques0[i0],)  # orig_c1
                rule_vals = {orig_c0: vals_orig0, orig_c1: vals_orig1}
                bit_masks = {orig_c0: int(mask1), orig_c1: int(mask0)}
                cols_out = (orig_c0, orig_c1)

            cand = RuleCandidate(
                profit_col=profit_col,
                cols=cols_out,
                bit_masks=bit_masks,
                rule_values=rule_vals,
                rule_text=_rule_text(rule_vals),
                n_rows=int(n_run),
                profit_sum=float(s_run),
                profit_mean=float(mean),
                winrate=float(winrate),
                stability_score=float(stab_score),
                timeline_bins=int(bins),
                timeline_coverage=float(cov),
                timeline_evenness=float(eve),
                timeline_top_share=float(top_share),
                timeline_pos_bins=int(pos_bins),
                timeline_neg_bins=int(neg_bins),
                dd_abs_bins=float(dd_abs),
                dd_rel_bins=float(dd_rel),
                complexity=float(complexity),
                score=float(score),
            )
            _push_topk(heap, cand, top_k=int(top_k))

    # ---- основной перебор: Gray по mask0 и mask1, без DP-таблиц на все mask0 ----
    vec_sum0 = np.zeros(k1, dtype=np.float64)
    vec_cnt0 = np.zeros(k1, dtype=np.int64)
    vec_pos0 = np.zeros(k1, dtype=np.int64)
    mat_bin_sum0 = np.zeros((bins, k1), dtype=np.float64)
    mat_bin_cnt0 = np.zeros((bins, k1), dtype=np.int64)

    prev_mask0 = 0
    bits0_on = 0

    bin_profit_run = np.empty(int(bins), dtype=np.float64)
    bin_count_run = np.empty(int(bins), dtype=np.int64)

    full0 = 1 << int(k0)
    full1 = 1 << int(k1)

    for i0 in range(1, full0):
        mask0 = i0 ^ (i0 >> 1)
        delta0 = mask0 ^ prev_mask0
        bit0 = int(delta0.bit_length() - 1)

        turned_on0 = ((mask0 >> bit0) & 1) == 1
        if turned_on0:
            bits0_on += 1
            vec_sum0 += sum_2d[bit0]
            vec_cnt0 += cnt_2d[bit0]
            vec_pos0 += pos_2d[bit0]
            mat_bin_sum0 += sum_bin_2d[:, bit0, :]
            mat_bin_cnt0 += cnt_bin_2d[:, bit0, :]
        else:
            bits0_on -= 1
            vec_sum0 -= sum_2d[bit0]
            vec_cnt0 -= cnt_2d[bit0]
            vec_pos0 -= pos_2d[bit0]
            mat_bin_sum0 -= sum_bin_2d[:, bit0, :]
            mat_bin_cnt0 -= cnt_bin_2d[:, bit0, :]

        prev_mask0 = mask0

        if float(bits0_on) > max_sel_thr0:
            _padd((full1 - 1))
            continue

        # если даже “все значения второй колонки” не набирают min_rows — можно пропустить весь mask0
        if int(vec_cnt0.sum()) < int(min_rows):
            _padd((full1 - 1))
            continue

        sel0_frac = float(bits0_on) * inv_k0

        prev_mask1 = 0
        bits1_on = 0

        n_run = 0
        s_run = 0.0
        pos_run = 0

        bin_profit_run.fill(0.0)
        bin_count_run.fill(0)

        for i1 in range(1, full1):
            _padd(1)

            mask1 = i1 ^ (i1 >> 1)
            delta1 = mask1 ^ prev_mask1
            bit1 = int(delta1.bit_length() - 1)

            turned_on1 = ((mask1 >> bit1) & 1) == 1
            if turned_on1:
                bits1_on += 1
                n_run += int(vec_cnt0[bit1])
                s_run += float(vec_sum0[bit1])
                pos_run += int(vec_pos0[bit1])
                bin_profit_run += mat_bin_sum0[:, bit1]
                bin_count_run += mat_bin_cnt0[:, bit1]
            else:
                bits1_on -= 1
                n_run -= int(vec_cnt0[bit1])
                s_run -= float(vec_sum0[bit1])
                pos_run -= int(vec_pos0[bit1])
                bin_profit_run -= mat_bin_sum0[:, bit1]
                bin_count_run -= mat_bin_cnt0[:, bit1]

            prev_mask1 = mask1

            if float(bits1_on) > max_sel_thr1:
                continue
            if n_run < int(min_rows):
                continue

            mean = float(s_run / n_run) if n_run > 0 else float("nan")
            if require_pos_mean and not (mean > 0.0):
                continue
            if mean < float(min_mean):
                continue

            winrate = float(pos_run / n_run) if n_run > 0 else 0.0
            if winrate < float(min_winrate):
                continue

            sel1_frac = float(bits1_on) * inv_k1
            complexity = float(0.5 * (sel0_frac + sel1_frac))

            if log1p_table is not None and 0 <= n_run < int(len(log1p_table)) and log1p_total > 0.0:
                rows_bonus = float(log1p_table[int(n_run)]) / float(log1p_total)
            else:
                rows_bonus = float(math.log1p(int(n_run)) / math.log1p(max(1, int(n_total_rows))))

            cplx_bonus = float(1.0 - complexity)
            mean_norm = float(mean) * inv_mean_scale

            base_score = (
                w_mean * mean_norm
                + w_win * float(winrate)
                + w_rows * float(rows_bonus)
                + w_cplx * float(cplx_bonus)
            )

            if can_bound and len(heap) >= int(top_k):
                worst = float(heap[0][0])
                ub = base_score + (w_stab * stab_best)
                if ub <= worst:
                    continue

            stab_score, cov, eve, top_share, pos_bins, neg_bins, dd_abs = _calc_stability_and_dd_fast(
                bin_profit=bin_profit_run,
                bin_count=bin_count_run,
                top_k_bins=top_k_bins,
                stab_w=stab_w,
            )

            dd_scale = max(1e-12, abs(float(s_run)))
            dd_rel = float(dd_abs / dd_scale)
            if max_dd_rel is not None and dd_rel > float(max_dd_rel):
                continue

            score = base_score + w_stab * float(stab_score) - w_dd * float(dd_rel)
            if len(heap) >= int(top_k) and score <= float(heap[0][0]):
                continue

            if not swapped:
                vals0 = _mask_to_values(mask0, uniques0)
                vals1 = _mask_to_values(mask1, uniques1)
                rule_vals = {c0: vals0, c1: vals1}
                bit_masks = {c0: int(mask0), c1: int(mask1)}
                cols_out = (c0, c1)
            else:
                # вернёмся к исходному порядку (orig_c0, orig_c1)
                vals_orig0 = _mask_to_values(mask1, uniques1)  # orig_c0
                vals_orig1 = _mask_to_values(mask0, uniques0)  # orig_c1
                rule_vals = {orig_c0: vals_orig0, orig_c1: vals_orig1}
                bit_masks = {orig_c0: int(mask1), orig_c1: int(mask0)}
                cols_out = (orig_c0, orig_c1)

            cand = RuleCandidate(
                profit_col=profit_col,
                cols=cols_out,
                bit_masks=bit_masks,
                rule_values=rule_vals,
                rule_text=_rule_text(rule_vals),
                n_rows=int(n_run),
                profit_sum=float(s_run),
                profit_mean=float(mean),
                winrate=float(winrate),
                stability_score=float(stab_score),
                timeline_bins=int(bins),
                timeline_coverage=float(cov),
                timeline_evenness=float(eve),
                timeline_top_share=float(top_share),
                timeline_pos_bins=int(pos_bins),
                timeline_neg_bins=int(neg_bins),
                dd_abs_bins=float(dd_abs),
                dd_rel_bins=float(dd_rel),
                complexity=float(complexity),
                score=float(score),
            )
            _push_topk(heap, cand, top_k=int(top_k))

    if progress is not None and acc > 0:
        progress.add(acc)

    out = [e[2] for e in sorted(heap, key=lambda x: x[0], reverse=True)]
    return out



def _search_3d(
    *,
    cols: Tuple[str, str, str],
    uniques_list: List[List[Any]],
    k_list: List[int],
    profit_col: str,
    sum_3d: np.ndarray,              # (k0,k1,k2)
    cnt_3d: np.ndarray,              # (k0,k1,k2)
    pos_3d: np.ndarray,              # (k0,k1,k2)
    sum_bin_3d: np.ndarray,          # (bins,k0,k1,k2)
    cnt_bin_3d: np.ndarray,          # (bins,k0,k1,k2)
    bins: int,
    top_frac: float,
    min_rows: int,
    max_sel_frac: float,
    require_pos_mean: bool,
    min_mean: float,
    min_winrate: float,
    max_dd_rel: Optional[float],
    obj: Dict[str, float],
    stab_w: Dict[str, float],
    n_total_rows: int,
    top_k: int,
    progress: Optional[_Progress],
    max_candidates_guard: Optional[int],   # <-- ВАЖНО: теперь Optional
    log1p_table: np.ndarray,
    log1p_total: float,
) -> List[RuleCandidate]:
    k0, k1, k2 = [int(x) for x in k_list]
    n_cand = ((1 << k0) - 1) * ((1 << k1) - 1) * ((1 << k2) - 1)

    # guard можно ОТКЛЮЧИТЬ, если max_candidates_guard <= 0
    if int(max_candidates_guard) > 0 and n_cand > int(max_candidates_guard):
        print(f"[WARN] 3D search skipped for cols={cols} because candidates={n_cand:,} > guard={int(max_candidates_guard):,}")
        return []

    c0, c1, c2n = cols

    u0, u1, u2 = uniques_list

    sum_3d = np.asarray(sum_3d, dtype=np.float64)
    cnt_3d = np.asarray(cnt_3d, dtype=np.int64)
    pos_3d = np.asarray(pos_3d, dtype=np.int64)
    sum_bin_3d = np.asarray(sum_bin_3d, dtype=np.float64)
    cnt_bin_3d = np.asarray(cnt_bin_3d, dtype=np.int64)

    mean_scale = float(obj.get("mean_scale", 1.0))
    inv_mean_scale = (1.0 / mean_scale) if mean_scale != 0.0 else 1.0

    w_mean = float(obj.get("w_mean", 0.0))
    w_win = float(obj.get("w_winrate", 0.0))
    w_stab = float(obj.get("w_stability", 0.0))
    w_dd = float(obj.get("w_drawdown", 0.0))
    w_rows = float(obj.get("w_rows", 0.0))
    w_cplx = float(obj.get("w_complexity", 0.0))

    stab_best = 1.5 if w_stab >= 0.0 else -1.0
    can_bound = (w_dd >= 0.0)

    inv_k0 = 1.0 / float(max(1, int(k0)))
    inv_k1 = 1.0 / float(max(1, int(k1)))
    inv_k2 = 1.0 / float(max(1, int(k2)))

    max_sel_thr0 = float(max_sel_frac) * float(k0)
    max_sel_thr1 = float(max_sel_frac) * float(k1)
    max_sel_thr2 = float(max_sel_frac) * float(k2)

    top_k_bins = int(max(1, math.ceil(float(bins) * float(top_frac))))

    # DP по mask0
    sum_m0 = np.zeros((1 << k0, k1, k2), dtype=np.float64)
    cnt_m0 = np.zeros((1 << k0, k1, k2), dtype=np.int64)
    pos_m0 = np.zeros((1 << k0, k1, k2), dtype=np.int64)

    sum_bin_m0 = np.zeros((1 << k0, bins, k1, k2), dtype=np.float64)
    cnt_bin_m0 = np.zeros((1 << k0, bins, k1, k2), dtype=np.int64)

    for mask0 in range(1, (1 << k0)):
        lsb = mask0 & -mask0
        bit = int(lsb.bit_length() - 1)
        prev = mask0 ^ lsb
        sum_m0[mask0] = sum_m0[prev] + sum_3d[bit]
        cnt_m0[mask0] = cnt_m0[prev] + cnt_3d[bit]
        pos_m0[mask0] = pos_m0[prev] + pos_3d[bit]
        sum_bin_m0[mask0] = sum_bin_m0[prev] + sum_bin_3d[:, bit, :, :]
        cnt_bin_m0[mask0] = cnt_bin_m0[prev] + cnt_bin_3d[:, bit, :, :]

    heap: List[Tuple[float, int, RuleCandidate]] = []
    upd_every = int(progress.update_every) if progress is not None else 0

    for mask0 in range(1, (1 << k0)):
        sel0_bits = _popcount(mask0)
        if float(sel0_bits) > max_sel_thr0:
            if progress is not None:
                progress.add(((1 << k1) - 1) * ((1 << k2) - 1))
            continue

        w2d = sum_m0[mask0]
        c2d = cnt_m0[mask0]
        p2d = pos_m0[mask0]
        wb2d = sum_bin_m0[mask0]
        cb2d = cnt_bin_m0[mask0]

        # DP по mask1
        sum_m1 = np.zeros((1 << k1, k2), dtype=np.float64)
        cnt_m1 = np.zeros((1 << k1, k2), dtype=np.int64)
        pos_m1 = np.zeros((1 << k1, k2), dtype=np.int64)

        sum_bin_m1 = np.zeros((1 << k1, bins, k2), dtype=np.float64)
        cnt_bin_m1 = np.zeros((1 << k1, bins, k2), dtype=np.int64)

        for mask1 in range(1, (1 << k1)):
            lsb = mask1 & -mask1
            bit = int(lsb.bit_length() - 1)
            prev = mask1 ^ lsb
            sum_m1[mask1] = sum_m1[prev] + w2d[bit]
            cnt_m1[mask1] = cnt_m1[prev] + c2d[bit]
            pos_m1[mask1] = pos_m1[prev] + p2d[bit]
            sum_bin_m1[mask1] = sum_bin_m1[prev] + wb2d[:, bit, :]
            cnt_bin_m1[mask1] = cnt_bin_m1[prev] + cb2d[:, bit, :]

        for mask1 in range(1, (1 << k1)):
            sel1_bits = _popcount(mask1)
            if float(sel1_bits) > max_sel_thr1:
                if progress is not None:
                    progress.add((1 << k2) - 1)
                continue

            vec_sum = sum_m1[mask1]
            vec_cnt = cnt_m1[mask1]
            vec_pos = pos_m1[mask1]
            mat_bin_sum = sum_bin_m1[mask1]
            mat_bin_cnt = cnt_bin_m1[mask1]

            prev_mask2 = 0
            bits2_on = 0

            n_run = 0
            s_run = 0.0
            pos_run = 0

            bin_profit_run = np.zeros(int(bins), dtype=np.float64)
            bin_count_run = np.zeros(int(bins), dtype=np.int64)

            acc = 0
            full2 = 1 << int(k2)
            for i2 in range(1, full2):
                if progress is not None:
                    acc += 1
                    if acc >= upd_every:
                        progress.add(acc)
                        acc = 0

                mask2 = i2 ^ (i2 >> 1)
                delta = mask2 ^ prev_mask2
                bit = int(delta.bit_length() - 1)

                turned_on = ((mask2 >> bit) & 1) == 1
                if turned_on:
                    bits2_on += 1
                    n_run += int(vec_cnt[bit])
                    s_run += float(vec_sum[bit])
                    pos_run += int(vec_pos[bit])
                    bin_profit_run += mat_bin_sum[:, bit]
                    bin_count_run += mat_bin_cnt[:, bit]
                else:
                    bits2_on -= 1
                    n_run -= int(vec_cnt[bit])
                    s_run -= float(vec_sum[bit])
                    pos_run -= int(vec_pos[bit])
                    bin_profit_run -= mat_bin_sum[:, bit]
                    bin_count_run -= mat_bin_cnt[:, bit]

                prev_mask2 = mask2

                if n_run < int(min_rows):
                    continue
                if float(bits2_on) > max_sel_thr2:
                    continue

                mean = float(s_run / n_run) if n_run > 0 else float("nan")
                if require_pos_mean and not (mean > 0.0):
                    continue
                if mean < float(min_mean):
                    continue

                winrate = float(pos_run / n_run) if n_run > 0 else 0.0
                if winrate < float(min_winrate):
                    continue

                sel0_frac = float(sel0_bits) * inv_k0
                sel1_frac = float(sel1_bits) * inv_k1
                sel2_frac = float(bits2_on) * inv_k2
                complexity = float((sel0_frac + sel1_frac + sel2_frac) / 3.0)

                rows_bonus = float(log1p_table[int(n_run)] / log1p_total) if log1p_total > 0.0 else 0.0
                cplx_bonus = float(1.0 - complexity)
                mean_norm = float(mean) * inv_mean_scale

                base_score = (
                    w_mean * mean_norm
                    + w_win * float(winrate)
                    + w_rows * rows_bonus
                    + w_cplx * cplx_bonus
                )

                if can_bound and len(heap) >= int(top_k):
                    worst = float(heap[0][0])
                    ub = base_score + (w_stab * stab_best)
                    if ub <= worst:
                        continue

                stab_score, cov, eve, top_share, pos_bins, neg_bins, dd_abs = _calc_stability_and_dd_fast(
                    bin_profit=bin_profit_run,
                    bin_count=bin_count_run,
                    top_k_bins=top_k_bins,
                    stab_w=stab_w,
                )

                dd_scale = max(1e-12, abs(float(s_run)))
                dd_rel = float(dd_abs / dd_scale)

                if max_dd_rel is not None and dd_rel > float(max_dd_rel):
                    continue

                score = base_score + w_stab * float(stab_score) - w_dd * float(dd_rel)
                if len(heap) >= int(top_k) and score <= float(heap[0][0]):
                    continue

                vals0 = _mask_to_values(mask0, u0)
                vals1 = _mask_to_values(mask1, u1)
                vals2 = _mask_to_values(mask2, u2)
                rule_vals = {c0: vals0, c1: vals1, c2n: vals2}
                rule_txt = _rule_text(rule_vals)

                cand = RuleCandidate(
                    profit_col=profit_col,
                    cols=(c0, c1, c2n),
                    bit_masks={c0: int(mask0), c1: int(mask1), c2n: int(mask2)},
                    rule_values=rule_vals,
                    rule_text=rule_txt,
                    n_rows=int(n_run),
                    profit_sum=float(s_run),
                    profit_mean=float(mean),
                    winrate=float(winrate),
                    stability_score=float(stab_score),
                    timeline_bins=int(bins),
                    timeline_coverage=float(cov),
                    timeline_evenness=float(eve),
                    timeline_top_share=float(top_share),
                    timeline_pos_bins=int(pos_bins),
                    timeline_neg_bins=int(neg_bins),
                    dd_abs_bins=float(dd_abs),
                    dd_rel_bins=float(dd_rel),
                    complexity=float(complexity),
                    score=float(score),
                )
                _push_topk(heap, cand, top_k=int(top_k))

            if progress is not None and acc > 0:
                progress.add(acc)

    out = [e[2] for e in sorted(heap, key=lambda x: x[0], reverse=True)]
    return out




# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def find_best_trade_rules(
    csv_path: str,
    *,
    max_var: int = MAX_VAR,
    profit_cols: Tuple[str, ...] = DEFAULT_PROFIT_COLS,
    exclude_cols: Tuple[str, ...] = DEFAULT_EXCLUDE_COLS,
    max_unique: int = MAX_UNIQUE,
    min_rows: int = DEFAULT_MIN_ROWS,
    show_progress: bool = True,
    print_results: bool = True,

    objective: Optional[Dict[str, float]] = None,
    stability_weights: Optional[Dict[str, float]] = None,
    timeline: Optional[Dict[str, Any]] = None,

    require_positive_mean: bool = DEFAULT_REQUIRE_POSITIVE_MEAN,
    min_mean: float = DEFAULT_MIN_MEAN,
    min_winrate: float = DEFAULT_MIN_WINRATE,
    max_drawdown_rel: Optional[float] = DEFAULT_MAX_DD_REL,
    max_selected_fraction_per_col: float = DEFAULT_MAX_SELECTED_FRACTION_PER_COL,

    top_k: int = DEFAULT_TOP_K,
    recalc_exact_dd_for_top: bool = DEFAULT_RECALC_EXACT_DD_FOR_TOP,

    # Guard 3D: можно выключить (None или <=0)
    max_candidates_guard_3d: Optional[int] = 3_000_000,

    # НОВОЕ: базовый фильтр (фиксированная “база”)
    base_filter: Optional[Dict[str, Any]] = None,
) -> Dict[Tuple[str, ...], Dict[str, List[RuleCandidate]]]:

    if not isinstance(max_var, int) or max_var < 1:
        raise ValueError("max_var must be an integer >= 1")
    if max_var > 3:
        raise ValueError("max_var up to 3 is supported")

    obj = dict(DEFAULT_OBJECTIVE)
    if objective:
        obj.update(objective)

    stab_w = dict(DEFAULT_STABILITY_WEIGHTS)
    if stability_weights:
        stab_w.update(stability_weights)

    tl = dict(DEFAULT_TIMELINE)
    if timeline:
        tl.update(timeline)

    bins = int(tl.get("bins", BINS))
    top_frac = float(tl.get("top_frac", 0.2))
    bin_mode = str(tl.get("bin_mode", "rows"))

    if bins < 2:
        raise ValueError("timeline bins must be >= 2")
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("timeline top_frac must be in (0,1]")

    df = pd.read_csv(csv_path)

    # 1) нормализация tm_ms
    if "tm_ms" in df.columns:
        df = df.copy()
        df["tm_ms"] = pd.to_numeric(df["tm_ms"], errors="coerce")

    # 2) фильтр по периоду
    start_dt = getattr(sv, "START", None)
    end_dt = getattr(sv, "END", None)

    if start_dt is not None or end_dt is not None:
        if "tm_ms" not in df.columns:
            raise ValueError("To filter by period you need 'tm_ms' column in CSV.")
        if df["tm_ms"].isna().all():
            raise ValueError("'tm_ms' could not be parsed as numbers.")

        start_ms = _dt_to_ms(start_dt) if start_dt is not None else None
        end_ms = _dt_to_ms(end_dt) if end_dt is not None else None

        if start_ms is not None:
            df = df[df["tm_ms"] >= int(start_ms)]
        if end_ms is not None:
            df = df[df["tm_ms"] < int(end_ms)]
        df = df.reset_index(drop=True)

    # 3) выкинуть строки, где profit_cols невалидны (NaN/inf)
    for pc in profit_cols:
        if pc not in df.columns:
            raise ValueError(f"Profit column '{pc}' not found in CSV.")

    valid = np.ones(len(df), dtype=np.bool_)
    for pc in profit_cols:
        arr = pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        valid &= np.isfinite(arr)

    if not valid.all():
        df = df[valid].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No rows left after filtering by period/profit validity.")

    # 4) сортировка по времени (если tm_ms есть)
    if "tm_ms" in df.columns:
        df = df.sort_values("tm_ms", kind="mergesort").reset_index(drop=True)

    # 4.5) НОВОЕ: применяем базовый фильтр (фиксированная база)
    # Если base_filter не передан — берём BASE_F (глобальную настройку)
    if base_filter is None:
        base_filter = globals().get("BASE_F", None)

    df_before_base = len(df)
    df, base_txt = _apply_base_filter(df, base_filter)
    df_after_base = len(df)

    if df_after_base == 0:
        raise ValueError(f"BASE_F removed all rows. base_filter={base_filter}")

    if print_results and base_txt:
        print()
        print("=" * 120)
        print("BASE FILTER (fixed condition applied before search)")
        print("=" * 120)
        kept_frac = (df_after_base / df_before_base) if df_before_base > 0 else 0.0
        print(f"Base: {base_txt}")
        print(f"Rows before base: {df_before_base:,}")
        print(f"Rows after base : {df_after_base:,} ({kept_frac*100:.2f}%)")
        print("=" * 120)

    # 5) признаки (базовые колонки НЕ участвуют в переборе)
    base_cols = set(base_filter.keys()) if base_filter else set()
    feature_cols = [c for c in df.columns if c not in profit_cols and c not in exclude_cols and c not in base_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions/base filter columns.")

    # 6) фильтр по уникальности и константам
    eligible_cols: List[str] = []
    skipped_cols: List[Tuple[str, int]] = []
    skipped_constant: List[Tuple[str, Any]] = []

    for c in feature_cols:
        s = df[c]
        non_na = s.dropna()
        if non_na.nunique() <= 1:
            const_val = non_na.iloc[0] if len(non_na) else None
            skipped_constant.append((c, const_val))
            continue

        nun = int(s.nunique(dropna=False))
        if nun <= int(max_unique):
            eligible_cols.append(c)
        else:
            skipped_cols.append((c, nun))

    if len(eligible_cols) < max_var:
        raise ValueError(f"Not enough eligible columns for max_var={max_var}. Eligible: {len(eligible_cols)}")

    # 6.5) печать: какие колонки участвуют и сколько уникальных
    if print_results:
        print()
        print("=" * 120)
        print("ELIGIBLE FEATURE COLUMNS (will be used in search)")
        print("=" * 120)

        info = []
        for c in eligible_cols:
            nun = int(df[c].nunique(dropna=False))
            n_non_na = int(df[c].notna().sum())
            info.append((c, nun, n_non_na))

        info.sort(key=lambda x: (-x[1], x[0]))

        for c, nun, n_non_na in info:
            print(f"  {c:<35} uniques={nun:<6} non_na={n_non_na:<8}")

        print("=" * 120)
        if skipped_cols:
            print("SKIPPED (too many uniques):")
            for c, nun in sorted(skipped_cols, key=lambda x: -x[1]):
                print(f"  {c:<35} uniques={nun}")
            print("=" * 120)

        if skipped_constant:
            print("SKIPPED (constant / nearly constant):")
            for c, v in sorted(skipped_constant, key=lambda x: x[0]):
                print(f"  {c:<35} const_value={v}")
            print("=" * 120)

    # 7) факторизация признаков
    fact = _factorize_columns(df, eligible_cols)

    # 8) биннинг (после BASE_F!)
    bin_id = _make_bin_id(df, bins=bins, bin_mode=bin_mode)

    # 9) наборы колонок длины max_var
    col_sets = list(itertools.combinations(eligible_cols, max_var))

    # 10) оценка веса прогресса
    def _cand_count_for_cols(cols: Tuple[str, ...]) -> int:
        ks = [int(fact[c]["k"]) for c in cols]
        total = 1
        for k in ks:
            total *= ((1 << k) - 1)
        return int(total)

    total_steps = 0
    for cs in col_sets:
        total_steps += _cand_count_for_cols(cs) * len(profit_cols)

    progress = _Progress(total=total_steps, enabled=show_progress)

    # профиты
    profit_arrays: Dict[str, np.ndarray] = {
        pc: pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        for pc in profit_cols
    }

    n_total_rows = int(len(df))
    log1p_table, log1p_total = _build_log1p_table(n_total_rows)

    results: Dict[Tuple[str, ...], Dict[str, List[RuleCandidate]]] = {}

    def _prepend_base_text_to_candidates(cands: List[RuleCandidate]) -> List[RuleCandidate]:
        if not base_txt:
            return cands
        out: List[RuleCandidate] = []
        for cand in cands:
            rt = f"{base_txt} and {cand.rule_text}" if cand.rule_text else base_txt
            out.append(RuleCandidate(
                profit_col=cand.profit_col,
                cols=cand.cols,
                bit_masks=cand.bit_masks,
                rule_values=cand.rule_values,
                rule_text=rt,
                n_rows=cand.n_rows,
                profit_sum=cand.profit_sum,
                profit_mean=cand.profit_mean,
                winrate=cand.winrate,
                stability_score=cand.stability_score,
                timeline_bins=cand.timeline_bins,
                timeline_coverage=cand.timeline_coverage,
                timeline_evenness=cand.timeline_evenness,
                timeline_top_share=cand.timeline_top_share,
                timeline_pos_bins=cand.timeline_pos_bins,
                timeline_neg_bins=cand.timeline_neg_bins,
                dd_abs_bins=cand.dd_abs_bins,
                dd_rel_bins=cand.dd_rel_bins,
                dd_abs_exact=cand.dd_abs_exact,
                dd_rel_exact=cand.dd_rel_exact,
                complexity=cand.complexity,
                score=cand.score,
            ))
        return out

    try:
        for cols in col_sets:
            ks = [int(fact[c]["k"]) for c in cols]
            uniques_list = [fact[c]["uniques"] for c in cols]
            codes_list = [fact[c]["codes"] for c in cols]
            shape = tuple(ks)

            flat_idx = np.ravel_multi_index(tuple(codes_list), dims=shape, mode="raise")
            size = int(np.prod(shape, dtype=np.int64))

            cnt_flat = _bincount_tensor(flat_idx, size=size, weights=None)
            cnt_tensor = cnt_flat.reshape(shape)

            cnt_bin_flat = _bincount_bin_tensor(bin_id, flat_idx, n_bins=bins, size=size, weights=None)
            cnt_bin_tensor = cnt_bin_flat.reshape((bins,) + shape)

            per_profit: Dict[str, List[RuleCandidate]] = {}

            for pc in profit_cols:
                p = profit_arrays[pc]

                sum_flat = _bincount_tensor(flat_idx, size=size, weights=p)
                sum_tensor = sum_flat.reshape(shape)

                pos_mask = (p > 0.0)
                pos_flat = np.bincount(flat_idx[pos_mask], minlength=size).astype(np.int64, copy=False)
                pos_tensor = pos_flat.reshape(shape)

                sum_bin_flat = _bincount_bin_tensor(bin_id, flat_idx, n_bins=bins, size=size, weights=p)
                sum_bin_tensor = sum_bin_flat.reshape((bins,) + shape)

                if max_var == 1:
                    c0 = cols[0]
                    k0 = ks[0]
                    u0 = uniques_list[0]

                    best = _search_1d(
                        col=c0,
                        uniques=u0,
                        k=k0,
                        profit_col=pc,
                        sum_by_val=np.asarray(sum_tensor, dtype=np.float64).reshape(k0),
                        cnt_by_val=np.asarray(cnt_tensor, dtype=np.int64).reshape(k0),
                        pos_by_val=np.asarray(pos_tensor, dtype=np.int64).reshape(k0),
                        sum_bin_by_val=np.asarray(sum_bin_tensor, dtype=np.float64).reshape(bins, k0),
                        cnt_bin_by_val=np.asarray(cnt_bin_tensor, dtype=np.int64).reshape(bins, k0),
                        bins=bins,
                        top_frac=top_frac,
                        min_rows=int(min_rows),
                        max_sel_frac=float(max_selected_fraction_per_col),
                        require_pos_mean=bool(require_positive_mean),
                        min_mean=float(min_mean),
                        min_winrate=float(min_winrate),
                        max_dd_rel=max_drawdown_rel,
                        obj=obj,
                        stab_w=stab_w,
                        n_total_rows=n_total_rows,
                        top_k=int(top_k),
                        progress=progress,
                        log1p_table=log1p_table,
                        log1p_total=log1p_total,
                    )
                    per_profit[pc] = _prepend_base_text_to_candidates(best)

                elif max_var == 2:
                    c0, c1 = cols
                    k0, k1 = ks
                    u0, u1 = uniques_list

                    best = _search_2d(
                        cols=(c0, c1),
                        uniques0=u0,
                        uniques1=u1,
                        k0=k0,
                        k1=k1,
                        profit_col=pc,
                        sum_2d=np.asarray(sum_tensor, dtype=np.float64).reshape(k0, k1),
                        cnt_2d=np.asarray(cnt_tensor, dtype=np.int64).reshape(k0, k1),
                        pos_2d=np.asarray(pos_tensor, dtype=np.int64).reshape(k0, k1),
                        sum_bin_2d=np.asarray(sum_bin_tensor, dtype=np.float64).reshape(bins, k0, k1),
                        cnt_bin_2d=np.asarray(cnt_bin_tensor, dtype=np.int64).reshape(bins, k0, k1),
                        bins=bins,
                        top_frac=top_frac,
                        min_rows=int(min_rows),
                        max_sel_frac=float(max_selected_fraction_per_col),
                        require_pos_mean=bool(require_positive_mean),
                        min_mean=float(min_mean),
                        min_winrate=float(min_winrate),
                        max_dd_rel=max_drawdown_rel,
                        obj=obj,
                        stab_w=stab_w,
                        n_total_rows=n_total_rows,
                        top_k=int(top_k),
                        progress=progress,
                        log1p_table=log1p_table,
                        log1p_total=log1p_total,
                    )
                    per_profit[pc] = _prepend_base_text_to_candidates(best)

                else:
                    c0, c1, c2 = cols

                    best = _search_3d(
                        cols=(c0, c1, c2),
                        uniques_list=uniques_list,     # type: ignore
                        k_list=ks,                     # type: ignore
                        profit_col=pc,
                        sum_3d=np.asarray(sum_tensor, dtype=np.float64).reshape(shape),
                        cnt_3d=np.asarray(cnt_tensor, dtype=np.int64).reshape(shape),
                        pos_3d=np.asarray(pos_tensor, dtype=np.int64).reshape(shape),
                        sum_bin_3d=np.asarray(sum_bin_tensor, dtype=np.float64).reshape((bins,) + shape),
                        cnt_bin_3d=np.asarray(cnt_bin_tensor, dtype=np.int64).reshape((bins,) + shape),
                        bins=bins,
                        top_frac=top_frac,
                        min_rows=int(min_rows),
                        max_sel_frac=float(max_selected_fraction_per_col),
                        require_pos_mean=bool(require_positive_mean),
                        min_mean=float(min_mean),
                        min_winrate=float(min_winrate),
                        max_dd_rel=max_drawdown_rel,
                        obj=obj,
                        stab_w=stab_w,
                        n_total_rows=n_total_rows,
                        top_k=int(top_k),
                        progress=progress,
                        max_candidates_guard=max_candidates_guard_3d,
                        log1p_table=log1p_table,
                        log1p_total=log1p_total,
                    )
                    per_profit[pc] = _prepend_base_text_to_candidates(best)

            results[tuple(cols)] = per_profit

        # точная просадка для top-k (на уже отфильтрованном df!)
        if recalc_exact_dd_for_top:
            for cols, per_profit in results.items():
                for pc, cands in per_profit.items():
                    if not cands:
                        continue

                    p = profit_arrays[pc]
                    for i in range(len(cands)):
                        cand = cands[i]
                        include = np.ones(len(df), dtype=np.bool_)

                        for c in cand.cols:
                            codes = fact[c]["codes"]
                            k = int(fact[c]["k"])
                            bm = int(cand.bit_masks[c])

                            allowed = np.zeros(k, dtype=np.bool_)
                            for bit in range(k):
                                if (bm >> bit) & 1:
                                    allowed[bit] = True

                            include &= allowed[codes]

                        dd_abs_exact = _calc_exact_drawdown_on_rows(p, include_mask=include)
                        scale = max(1e-12, abs(float(p[include].sum())))
                        dd_rel_exact = float(dd_abs_exact / scale)

                        cands[i] = RuleCandidate(
                            profit_col=cand.profit_col,
                            cols=cand.cols,
                            bit_masks=cand.bit_masks,
                            rule_values=cand.rule_values,
                            rule_text=cand.rule_text,
                            n_rows=cand.n_rows,
                            profit_sum=cand.profit_sum,
                            profit_mean=cand.profit_mean,
                            winrate=cand.winrate,
                            stability_score=cand.stability_score,
                            timeline_bins=cand.timeline_bins,
                            timeline_coverage=cand.timeline_coverage,
                            timeline_evenness=cand.timeline_evenness,
                            timeline_top_share=cand.timeline_top_share,
                            timeline_pos_bins=cand.timeline_pos_bins,
                            timeline_neg_bins=cand.timeline_neg_bins,
                            dd_abs_bins=cand.dd_abs_bins,
                            dd_rel_bins=cand.dd_rel_bins,
                            dd_abs_exact=float(dd_abs_exact),
                            dd_rel_exact=float(dd_rel_exact),
                            complexity=cand.complexity,
                            score=cand.score,
                        )

        if print_results:
            print()
            print("=" * 120)
            print("RESULTS (multi-objective: mean + winrate + stability - drawdown)")
            print("=" * 120)
            print(f"CSV: {csv_path}")
            print(f"Rows used: {len(df):,}")
            if base_txt:
                print(f"BASE_F: {base_txt}")
            print(f"max_var={max_var} | min_rows={min_rows} | max_unique={max_unique}")
            print(f"Timeline: bins={bins} | bin_mode={bin_mode} | top_frac={top_frac}")
            print(f"Filters: require_pos_mean={require_positive_mean} | min_mean={min_mean} | min_winrate={min_winrate} | "
                  f"max_dd_rel={max_drawdown_rel} | max_selected_fraction_per_col={max_selected_fraction_per_col}")
            print("Objective weights:", obj)
            print("Stability weights:", stab_w)
            print("=" * 120)

            sort_pc = "profit_2" if "profit_2" in profit_cols else profit_cols[0]

            def _best_score_for(cs: Tuple[str, ...]) -> float:
                per = results.get(cs, {})
                cands = per.get(sort_pc, [])
                return float(cands[0].score) if cands else float("-inf")

            # ЛУЧШИЕ сверху
            sorted_sets = sorted(results.keys(), key=_best_score_for, reverse=True)

            for cs in sorted_sets:
                print("-" * 120)
                print(f"COLUMNS: {', '.join(cs)}")

                per = results[cs]
                for pc in profit_cols:
                    cands = per.get(pc, [])
                    if not cands:
                        print(f"  [{pc}] no candidates passed filters")
                        continue

                    print(f"  [{pc}] TOP-{min(top_k, len(cands))}:")
                    for rank, cand in enumerate(cands[:top_k], start=1):
                        dd_exact = cand.dd_rel_exact if cand.dd_rel_exact is not None else float("nan")
                        dd_exact_abs = cand.dd_abs_exact if cand.dd_abs_exact is not None else float("nan")

                        print(f"    #{rank} score={cand.score:.6f} | N={cand.n_rows} | mean={cand.profit_mean:.6f} | "
                              f"winrate={cand.winrate:.3f} | stab={cand.stability_score:.3f} | "
                              f"dd_rel_bins={cand.dd_rel_bins:.3f} | dd_rel_exact={dd_exact:.3f}")

                        print(f"        sum={cand.profit_sum:.6f} | complexity={cand.complexity:.3f}")
                        print(f"        timeline: coverage={cand.timeline_coverage:.3f} | evenness={cand.timeline_evenness:.3f} | "
                              f"top_share={cand.timeline_top_share:.3f} | pos_bins={cand.timeline_pos_bins} | neg_bins={cand.timeline_neg_bins}")
                        print(f"        dd_abs_bins={cand.dd_abs_bins:.6f} | dd_abs_exact={dd_exact_abs:.6f}")
                        print(f"        rule: {cand.rule_text}")

            print("-" * 120)

        return results

    finally:
        progress.close()




# Чтобы старый код не ломался, оставим имя как раньше
def find_best_profit_rules(
    csv_path: str,
    *,
    max_var: int = MAX_VAR,
    profit_cols: Tuple[str, ...] = DEFAULT_PROFIT_COLS,
    exclude_cols: Tuple[str, ...] = DEFAULT_EXCLUDE_COLS,
    max_unique: int = MAX_UNIQUE,
    min_rows: int = DEFAULT_MIN_ROWS,
    show_progress: bool = True,
    print_results: bool = True,
    objective: Optional[Dict[str, float]] = None,
    stability_weights: Optional[Dict[str, float]] = None,
    timeline: Optional[Dict[str, Any]] = None,
    require_positive_mean: bool = DEFAULT_REQUIRE_POSITIVE_MEAN,
    min_mean: float = DEFAULT_MIN_MEAN,
    min_winrate: float = DEFAULT_MIN_WINRATE,
    max_drawdown_rel: Optional[float] = DEFAULT_MAX_DD_REL,
    max_selected_fraction_per_col: float = DEFAULT_MAX_SELECTED_FRACTION_PER_COL,
    top_k: int = DEFAULT_TOP_K,
    recalc_exact_dd_for_top: bool = DEFAULT_RECALC_EXACT_DD_FOR_TOP,
    max_candidates_guard_3d: int = 3_000_000,
) -> Dict[Tuple[str, ...], Dict[str, List[RuleCandidate]]]:
    return find_best_trade_rules(
        csv_path=csv_path,
        max_var=max_var,
        profit_cols=profit_cols,
        exclude_cols=exclude_cols,
        max_unique=max_unique,
        min_rows=min_rows,
        show_progress=show_progress,
        print_results=print_results,
        objective=objective,
        stability_weights=stability_weights,
        timeline=timeline,
        require_positive_mean=require_positive_mean,
        min_mean=min_mean,
        min_winrate=min_winrate,
        max_drawdown_rel=max_drawdown_rel,
        max_selected_fraction_per_col=max_selected_fraction_per_col,
        top_k=top_k,
        recalc_exact_dd_for_top=recalc_exact_dd_for_top,
        max_candidates_guard_3d=max_candidates_guard_3d,
    )


# ======================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ======================================================================

def example_run() -> None:
    _ = find_best_trade_rules(
        csv_path=CSV_PATH,
        max_var=MAX_VAR,
        profit_cols=DEFAULT_PROFIT_COLS,
        exclude_cols=DEFAULT_EXCLUDE_COLS,
        max_unique=MAX_UNIQUE,
        min_rows=DEFAULT_MIN_ROWS,
        show_progress=True,
        print_results=True,

        # Можно менять веса “цели”
        objective={
            "mean_scale": 0.01,
            "w_mean": 1.0,
            "w_winrate": 0.7,
            "w_stability": 0.8,
            "w_drawdown": 1.0,
            "w_rows": 0.1,
            "w_complexity": 0.25,
        },

        # Можно менять как считается стабильность
        stability_weights={
            "w_coverage": 0.35,
            "w_evenness": 0.45,
            "w_pos_bins": 0.30,
            "w_top_share": 0.40,
            "w_neg_bins": 0.25,
        },

        # Таймлайн
        timeline={
            "bins": BINS,
            "top_frac": 0.2,
            "bin_mode": "rows",   # "rows" или "time"
        },

        # Фильтры качества
        require_positive_mean=False,
        min_mean=0.0,
        min_winrate=0.0,
        max_drawdown_rel=None,
        max_selected_fraction_per_col=0.95,
        max_candidates_guard_3d=0,

        # Выводить топ-N
        top_k=3,

        # Пересчитать точную просадку по реальным сделкам для top_k
        recalc_exact_dd_for_top=True,
    )


if __name__ == "__main__":
    # По вашему правилу: без CLI/argparse.
    example_run()
