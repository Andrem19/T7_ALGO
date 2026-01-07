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
MAX_VAR = 2

# Максимальная уникальность столбца, чтобы столбец попадал в перебор
MAX_UNIQUE = 24

# Сколько тайм-бинов для стабильности и просадки (больше = точнее, но тяжелее)
BINS = 12

# Какие столбцы нельзя использовать как признаки
DEFAULT_EXCLUDE_COLS = (
    "tm_ms",
    "feer_and_greed", "fg_stock", "feer_and_greed", "cls_15m", "cls_5m", "cls_30m", "cls_1h", "super_cls"
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
DEFAULT_MIN_ROWS = 50

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
    vals: List[Any] = []
    for i in range(len(uniques)):
        if (mask >> i) & 1:
            vals.append(uniques[i])
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
    строим кумулятивную прибыль по бинам, держим максимум (пик),
    и считаем наихудшее падение от пика до последующего значения.
    """
    peak = 0.0
    cur = 0.0
    max_dd = 0.0
    for x in bin_profit.tolist():
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
) -> float:
    mean_scale = float(obj.get("mean_scale", 1.0))
    mean_norm = float(profit_mean / mean_scale) if mean_scale != 0 else float(profit_mean)

    # Бонус за количество строк: чем ближе к общему объёму, тем выше бонус
    # (но это только слабый вес)
    if n_total_rows > 0:
        rows_bonus = float(math.log1p(n_rows) / math.log1p(n_total_rows))
    else:
        rows_bonus = 0.0

    # Complexity: это доля выбранных значений (0..1). Меньше — лучше.
    complexity_bonus = float(1.0 - max(0.0, min(1.0, complexity)))

    score = (
        float(obj.get("w_mean", 0.0)) * mean_norm
        + float(obj.get("w_winrate", 0.0)) * float(winrate)
        + float(obj.get("w_stability", 0.0)) * float(stability_score)
        - float(obj.get("w_drawdown", 0.0)) * float(dd_rel)
        + float(obj.get("w_rows", 0.0)) * rows_bonus
        + float(obj.get("w_complexity", 0.0)) * complexity_bonus
    )
    return float(score)


def _calc_exact_drawdown_on_rows(profit: np.ndarray, include_mask: np.ndarray) -> float:
    """
    Точная просадка на уровне сделок (строк).
    Идём по строкам в хронологическом порядке, учитываем только выбранные сделки,
    считаем худшее падение от пика к последующему значению.
    """
    peak = 0.0
    cur = 0.0
    max_dd = 0.0

    p = profit[include_mask]
    for x in p.tolist():
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
    sum_by_val: np.ndarray,          # (k,)
    cnt_by_val: np.ndarray,          # (k,)
    pos_by_val: np.ndarray,          # (k,)
    sum_bin_by_val: np.ndarray,      # (bins,k)
    cnt_bin_by_val: np.ndarray,      # (bins,k)
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
) -> List[RuleCandidate]:
    # Предвычисляем subset sums по маскам
    ss_sum = _subset_sums_1d(sum_by_val.astype(np.float64, copy=False))
    ss_cnt = _subset_sums_1d_int(cnt_by_val.astype(np.int64, copy=False))
    ss_pos = _subset_sums_1d_int(pos_by_val.astype(np.int64, copy=False))

    # Для бинов удобно делать DP по маскам отдельно для каждого бина
    # (bins небольшие, k небольшое)
    ss_bin_sum = np.zeros((1 << k, bins), dtype=np.float64)
    ss_bin_cnt = np.zeros((1 << k, bins), dtype=np.int64)

    for mask in range(1, (1 << k)):
        lsb = mask & -mask
        bit = int(lsb.bit_length() - 1)
        prev = mask ^ lsb
        ss_bin_sum[mask] = ss_bin_sum[prev] + sum_bin_by_val[:, bit]
        ss_bin_cnt[mask] = ss_bin_cnt[prev] + cnt_bin_by_val[:, bit]

    # индексы для печати/значений
    idx_list = _mask_indices_list(k)

    heap: List[Tuple[float, int, RuleCandidate]] = []

    for mask in range(1, (1 << k)):
        if progress is not None:
            progress.add(1)

        n = int(ss_cnt[mask])
        if n < min_rows:
            continue

        sel_frac = float(_popcount(mask) / max(1, k))
        if sel_frac > float(max_sel_frac):
            continue

        s = float(ss_sum[mask])
        mean = float(s / n) if n > 0 else float("nan")
        if require_pos_mean and not (mean > 0.0):
            continue
        if mean < float(min_mean):
            continue

        pos = int(ss_pos[mask])
        winrate = float(pos / n) if n > 0 else 0.0
        if winrate < float(min_winrate):
            continue

        bin_profit = ss_bin_sum[mask]          # (bins,)
        bin_count = ss_bin_cnt[mask]          # (bins,)

        stab_score, cov, eve, top_share, pos_bins, neg_bins = _calc_stability(
            bin_profit=bin_profit,
            bin_count=bin_count,
            top_frac=float(top_frac),
            stab_w=stab_w,
        )

        dd_abs = _calc_drawdown_from_bins(bin_profit)
        dd_scale = max(1e-12, abs(s))
        dd_rel = float(dd_abs / dd_scale)

        if max_dd_rel is not None and dd_rel > float(max_dd_rel):
            continue

        complexity = sel_frac
        score = _score_candidate(
            profit_mean=mean,
            winrate=winrate,
            stability_score=stab_score,
            dd_rel=dd_rel,
            n_rows=n,
            n_total_rows=n_total_rows,
            complexity=complexity,
            obj=obj,
        )

        vals = _mask_to_values(mask, uniques)
        rule_vals = {col: vals}
        rule_txt = _rule_text(rule_vals)

        cand = RuleCandidate(
            profit_col=profit_col,
            cols=(col,),
            bit_masks={col: int(mask)},
            rule_values=rule_vals,
            rule_text=rule_txt,
            n_rows=n,
            profit_sum=s,
            profit_mean=mean,
            winrate=winrate,
            stability_score=stab_score,
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
        _push_topk(heap, cand, top_k=top_k)

    # отсортировать по score убыванию
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
) -> List[RuleCandidate]:
    c0, c1 = cols

    # DP по маскам на оси 0: для каждого mask0 получаем вектор по оси 1
    sum_m0 = np.zeros((1 << k0, k1), dtype=np.float64)
    cnt_m0 = np.zeros((1 << k0, k1), dtype=np.int64)
    pos_m0 = np.zeros((1 << k0, k1), dtype=np.int64)

    # по бинам
    sum_bin_m0 = np.zeros((1 << k0, bins, k1), dtype=np.float64)
    cnt_bin_m0 = np.zeros((1 << k0, bins, k1), dtype=np.int64)

    for mask0 in range(1, (1 << k0)):
        lsb = mask0 & -mask0
        bit = int(lsb.bit_length() - 1)
        prev = mask0 ^ lsb
        sum_m0[mask0] = sum_m0[prev] + sum_2d[bit]
        cnt_m0[mask0] = cnt_m0[prev] + cnt_2d[bit]
        pos_m0[mask0] = pos_m0[prev] + pos_2d[bit]

        sum_bin_m0[mask0] = sum_bin_m0[prev] + sum_bin_2d[:, bit, :]
        cnt_bin_m0[mask0] = cnt_bin_m0[prev] + cnt_bin_2d[:, bit, :]

    idx_list_1 = _mask_indices_list(k1)

    heap: List[Tuple[float, int, RuleCandidate]] = []

    for mask0 in range(1, (1 << k0)):
        sel0_frac = float(_popcount(mask0) / max(1, k0))
        if sel0_frac > float(max_sel_frac):
            # всё равно надо делать progress, иначе он “зависнет” по ETA
            # но здесь мы не знаем, сколько mask1 будет пропущено; считаем 1 шаг за mask0
            if progress is not None:
                progress.add((1 << k1) - 1)
            continue

        vec_sum = sum_m0[mask0]           # (k1,)
        vec_cnt = cnt_m0[mask0]           # (k1,)
        vec_pos = pos_m0[mask0]           # (k1,)
        mat_bin_sum = sum_bin_m0[mask0]   # (bins,k1)
        mat_bin_cnt = cnt_bin_m0[mask0]   # (bins,k1)

        for mask1 in range(1, (1 << k1)):
            if progress is not None:
                progress.add(1)

            sel1_frac = float(_popcount(mask1) / max(1, k1))
            if sel1_frac > float(max_sel_frac):
                continue

            idx1 = idx_list_1[mask1]
            n = int(vec_cnt[idx1].sum())
            if n < min_rows:
                continue

            s = float(vec_sum[idx1].sum())
            mean = float(s / n) if n > 0 else float("nan")
            if require_pos_mean and not (mean > 0.0):
                continue
            if mean < float(min_mean):
                continue

            pos = int(vec_pos[idx1].sum())
            winrate = float(pos / n) if n > 0 else 0.0
            if winrate < float(min_winrate):
                continue

            bin_profit = mat_bin_sum[:, idx1].sum(axis=1)
            bin_count = mat_bin_cnt[:, idx1].sum(axis=1)

            stab_score, cov, eve, top_share, pos_bins, neg_bins = _calc_stability(
                bin_profit=bin_profit,
                bin_count=bin_count,
                top_frac=float(top_frac),
                stab_w=stab_w,
            )

            dd_abs = _calc_drawdown_from_bins(bin_profit)
            dd_scale = max(1e-12, abs(s))
            dd_rel = float(dd_abs / dd_scale)

            if max_dd_rel is not None and dd_rel > float(max_dd_rel):
                continue

            complexity = float(0.5 * (sel0_frac + sel1_frac))
            score = _score_candidate(
                profit_mean=mean,
                winrate=winrate,
                stability_score=stab_score,
                dd_rel=dd_rel,
                n_rows=n,
                n_total_rows=n_total_rows,
                complexity=complexity,
                obj=obj,
            )

            vals0 = _mask_to_values(mask0, uniques0)
            vals1 = _mask_to_values(mask1, uniques1)
            rule_vals = {c0: vals0, c1: vals1}
            rule_txt = _rule_text(rule_vals)

            cand = RuleCandidate(
                profit_col=profit_col,
                cols=(c0, c1),
                bit_masks={c0: int(mask0), c1: int(mask1)},
                rule_values=rule_vals,
                rule_text=rule_txt,
                n_rows=n,
                profit_sum=s,
                profit_mean=mean,
                winrate=winrate,
                stability_score=stab_score,
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
            _push_topk(heap, cand, top_k=top_k)

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
    max_candidates_guard: int,
) -> List[RuleCandidate]:
    # Guard: если слишком много комбинаций — лучше не убивать машину
    k0, k1, k2 = k_list
    n_cand = ((1 << k0) - 1) * ((1 << k1) - 1) * ((1 << k2) - 1)
    if n_cand > max_candidates_guard:
        print(f"[WARN] 3D search skipped for cols={cols} because candidates={n_cand:,} > guard={max_candidates_guard:,}")
        return []

    c0, c1, c2n = cols
    u0, u1, u2 = uniques_list

    # DP по mask0: сворачиваем ось 0 -> получаем 2D (k1,k2)
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

    idx_list_1 = _mask_indices_list(k1)
    idx_list_2 = _mask_indices_list(k2)

    heap: List[Tuple[float, int, RuleCandidate]] = []

    for mask0 in range(1, (1 << k0)):
        sel0_frac = float(_popcount(mask0) / max(1, k0))
        if sel0_frac > float(max_sel_frac):
            # прогресс приблизительно — считаем что мы “пропустили” весь внутренний перебор
            if progress is not None:
                progress.add(((1 << k1) - 1) * ((1 << k2) - 1))
            continue

        w2d = sum_m0[mask0]           # (k1,k2)
        c2d = cnt_m0[mask0]
        p2d = pos_m0[mask0]
        wb2d = sum_bin_m0[mask0]      # (bins,k1,k2)
        cb2d = cnt_bin_m0[mask0]

        # DP по mask1: сворачиваем ось 1 -> получаем вектор по оси 2
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
            if progress is not None:
                progress.add((1 << k2) - 1)

            sel1_frac = float(_popcount(mask1) / max(1, k1))
            if sel1_frac > float(max_sel_frac):
                continue

            vec_sum = sum_m1[mask1]
            vec_cnt = cnt_m1[mask1]
            vec_pos = pos_m1[mask1]
            mat_bin_sum = sum_bin_m1[mask1]   # (bins,k2)
            mat_bin_cnt = cnt_bin_m1[mask1]

            for mask2 in range(1, (1 << k2)):
                sel2_frac = float(_popcount(mask2) / max(1, k2))
                if sel2_frac > float(max_sel_frac):
                    continue

                idx2 = idx_list_2[mask2]
                n = int(vec_cnt[idx2].sum())
                if n < min_rows:
                    continue

                s = float(vec_sum[idx2].sum())
                mean = float(s / n) if n > 0 else float("nan")
                if require_pos_mean and not (mean > 0.0):
                    continue
                if mean < float(min_mean):
                    continue

                pos = int(vec_pos[idx2].sum())
                winrate = float(pos / n) if n > 0 else 0.0
                if winrate < float(min_winrate):
                    continue

                bin_profit = mat_bin_sum[:, idx2].sum(axis=1)
                bin_count = mat_bin_cnt[:, idx2].sum(axis=1)

                stab_score, cov, eve, top_share, pos_bins, neg_bins = _calc_stability(
                    bin_profit=bin_profit,
                    bin_count=bin_count,
                    top_frac=float(top_frac),
                    stab_w=stab_w,
                )

                dd_abs = _calc_drawdown_from_bins(bin_profit)
                dd_scale = max(1e-12, abs(s))
                dd_rel = float(dd_abs / dd_scale)

                if max_dd_rel is not None and dd_rel > float(max_dd_rel):
                    continue

                complexity = float((sel0_frac + sel1_frac + sel2_frac) / 3.0)
                score = _score_candidate(
                    profit_mean=mean,
                    winrate=winrate,
                    stability_score=stab_score,
                    dd_rel=dd_rel,
                    n_rows=n,
                    n_total_rows=n_total_rows,
                    complexity=complexity,
                    obj=obj,
                )

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
                    n_rows=n,
                    profit_sum=s,
                    profit_mean=mean,
                    winrate=winrate,
                    stability_score=stab_score,
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
                _push_topk(heap, cand, top_k=top_k)

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

    # Мульти-цель (веса) / стабильность / таймлайн
    objective: Optional[Dict[str, float]] = None,
    stability_weights: Optional[Dict[str, float]] = None,
    timeline: Optional[Dict[str, Any]] = None,

    # Фильтры качества
    require_positive_mean: bool = DEFAULT_REQUIRE_POSITIVE_MEAN,
    min_mean: float = DEFAULT_MIN_MEAN,
    min_winrate: float = DEFAULT_MIN_WINRATE,
    max_drawdown_rel: Optional[float] = DEFAULT_MAX_DD_REL,
    max_selected_fraction_per_col: float = DEFAULT_MAX_SELECTED_FRACTION_PER_COL,

    # Сколько лучших кандидатов сохранять на каждый набор колонок и profit_col
    top_k: int = DEFAULT_TOP_K,

    # Пересчитать точную просадку по строкам для top_k (уже после поиска)
    recalc_exact_dd_for_top: bool = DEFAULT_RECALC_EXACT_DD_FOR_TOP,

    # Защита от убийства CPU при 3D
    max_candidates_guard_3d: int = 3_000_000,
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

    # 5) признаки
    feature_cols = [c for c in df.columns if c not in profit_cols and c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions.")

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

    # 7) факторизация признаков
    fact = _factorize_columns(df, eligible_cols)

    # 8) биннинг по времени / строкам (для стабильности и просадки)
    bin_id = _make_bin_id(df, bins=bins, bin_mode=bin_mode)

    # 9) все наборы колонок фиксированной длины max_var
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

    # подготовка профитов
    profit_arrays: Dict[str, np.ndarray] = {
        pc: pd.to_numeric(df[pc], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        for pc in profit_cols
    }

    n_total_rows = int(len(df))
    results: Dict[Tuple[str, ...], Dict[str, List[RuleCandidate]]] = {}

    # ==================================================================
    # Перебор по наборам колонок
    # ==================================================================
    try:
        for cols in col_sets:
            ks = [int(fact[c]["k"]) for c in cols]
            uniques_list = [fact[c]["uniques"] for c in cols]
            codes_list = [fact[c]["codes"] for c in cols]
            shape = tuple(ks)

            # flat index по комбинациям категорий
            flat_idx = np.ravel_multi_index(tuple(codes_list), dims=shape, mode="raise")
            size = int(np.prod(shape, dtype=np.int64))

            # counts общий (не зависит от profit_col)
            cnt_flat = _bincount_tensor(flat_idx, size=size, weights=None)  # int64
            cnt_tensor = cnt_flat.reshape(shape)

            # counts по бинам общий
            cnt_bin_flat = _bincount_bin_tensor(bin_id, flat_idx, n_bins=bins, size=size, weights=None)  # (bins,size)
            cnt_bin_tensor = cnt_bin_flat.reshape((bins,) + shape)

            per_profit: Dict[str, List[RuleCandidate]] = {}

            for pc in profit_cols:
                p = profit_arrays[pc]

                # суммарная прибыль по ячейкам
                sum_flat = _bincount_tensor(flat_idx, size=size, weights=p)  # float64
                sum_tensor = sum_flat.reshape(shape)

                # “плюсовые сделки” по ячейкам
                pos_w = (p > 0.0).astype(np.int64, copy=False)
                pos_flat = _bincount_tensor(flat_idx, size=size, weights=pos_w.astype(np.float64, copy=False))
                pos_tensor = pos_flat.reshape(shape).astype(np.int64, copy=False)

                # суммарная прибыль по бинам
                sum_bin_flat = _bincount_bin_tensor(bin_id, flat_idx, n_bins=bins, size=size, weights=p)
                sum_bin_tensor = sum_bin_flat.reshape((bins,) + shape)

                # ----------------------------------------------------------
                # Поиск лучших кандидатов по новой цели
                # ----------------------------------------------------------
                if max_var == 1:
                    c0 = cols[0]
                    k0 = ks[0]
                    u0 = uniques_list[0]

                    sum_by_val = np.asarray(sum_tensor, dtype=np.float64).reshape(k0)
                    cnt_by_val = np.asarray(cnt_tensor, dtype=np.int64).reshape(k0)
                    pos_by_val = np.asarray(pos_tensor, dtype=np.int64).reshape(k0)

                    sum_bin_by_val = np.asarray(sum_bin_tensor, dtype=np.float64).reshape(bins, k0)
                    cnt_bin_by_val = np.asarray(cnt_bin_tensor, dtype=np.int64).reshape(bins, k0)

                    best = _search_1d(
                        col=c0,
                        uniques=u0,
                        k=k0,
                        profit_col=pc,
                        sum_by_val=sum_by_val,
                        cnt_by_val=cnt_by_val,
                        pos_by_val=pos_by_val,
                        sum_bin_by_val=sum_bin_by_val,
                        cnt_bin_by_val=cnt_bin_by_val,
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
                    )
                    per_profit[pc] = best

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
                    )
                    per_profit[pc] = best

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
                        max_candidates_guard=int(max_candidates_guard_3d),
                    )
                    per_profit[pc] = best

            results[tuple(cols)] = per_profit

        # --------------------------------------------------------------
        # Пересчёт точной просадки (по реальным сделкам) для TOP-K
        # --------------------------------------------------------------
        if recalc_exact_dd_for_top:
            for cols, per_profit in results.items():
                for pc, cands in per_profit.items():
                    if not cands:
                        continue

                    p = profit_arrays[pc]
                    # для каждого кандидата пересчитываем include_mask и точную просадку
                    for i in range(len(cands)):
                        cand = cands[i]
                        include = np.ones(len(df), dtype=np.bool_)
                        for c in cand.cols:
                            codes = fact[c]["codes"]
                            k = int(fact[c]["k"])
                            bm = int(cand.bit_masks[c])

                            # делаем allowed маску по категориям
                            allowed = np.zeros(k, dtype=np.bool_)
                            for bit in range(k):
                                if (bm >> bit) & 1:
                                    allowed[bit] = True
                            include &= allowed[codes]

                        dd_abs_exact = _calc_exact_drawdown_on_rows(p, include_mask=include)
                        scale = max(1e-12, abs(float(p[include].sum())))
                        dd_rel_exact = float(dd_abs_exact / scale)

                        # заменить объект на новый (dataclass frozen)
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

        # --------------------------------------------------------------
        # Печать
        # --------------------------------------------------------------
        if print_results:
            print()
            print("=" * 120)
            print("RESULTS (multi-objective: mean + winrate + stability - drawdown)")
            print("=" * 120)
            print(f"CSV: {csv_path}")
            print(f"Rows used: {len(df):,}")
            print(f"max_var={max_var} | min_rows={min_rows} | max_unique={max_unique}")
            print(f"Timeline: bins={bins} | bin_mode={bin_mode} | top_frac={top_frac}")
            print(f"Filters: require_pos_mean={require_positive_mean} | min_mean={min_mean} | min_winrate={min_winrate} | "
                  f"max_dd_rel={max_drawdown_rel} | max_selected_fraction_per_col={max_selected_fraction_per_col}")
            print("Objective weights:", obj)
            print("Stability weights:", stab_w)
            print("=" * 120)

            # Упорядочим наборы колонок по лучшему score (по profit_2 если есть, иначе по первому)
            sort_pc = "profit_2" if "profit_2" in profit_cols else profit_cols[0]

            def _best_score_for(cs: Tuple[str, ...]) -> float:
                per = results.get(cs, {})
                cands = per.get(sort_pc, [])
                return float(cands[0].score) if cands else float("-inf")

            sorted_sets = sorted(results.keys(), key=_best_score_for, reverse=False)


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
        require_positive_mean=True,
        min_mean=0.0,
        min_winrate=0.0,
        max_drawdown_rel=None,
        max_selected_fraction_per_col=0.85,

        # Выводить топ-N
        top_k=3,

        # Пересчитать точную просадку по реальным сделкам для top_k
        recalc_exact_dd_for_top=True,
    )


if __name__ == "__main__":
    # По вашему правилу: без CLI/argparse.
    example_run()
