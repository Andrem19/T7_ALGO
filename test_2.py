# combo_profit_search.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
import itertools
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import shared_vars as sv



# -----------------------------
# Настройки по умолчанию
# -----------------------------


CSV_PATH = "vector_6.csv"
MAX_VAR = 1
MAX_UNIQUE = 7
BINS = 12
DEFAULT_EXCLUDE_COLS = ("tm_ms", "h", "feer_and_greed", "fg_stock", "feer_and_greed", "rsi_1", "sp500", "atr_1", "iv_est_1", "squize_index_1", "vix")
DEFAULT_PROFIT_COLS = ("profit_1", "profit_2")

sv.START = datetime(2022, 1, 1)
sv.END = datetime(2025, 1, 1)

# Защита от столбцов с огромной уникальностью (иначе перебор подмножеств становится нереальным)
DEFAULT_MAX_UNIQUE = 12

# Минимум строк в отфильтрованном наборе (чтобы не выбирать правило на 1–2 строках)
DEFAULT_MIN_ROWS = 10

# Как часто обновлять прогресс (чтобы tqdm не тормозил на миллионах итераций)
DEFAULT_PROGRESS_UPDATE_EVERY = 2048

# По умолчанию: сколько бинов по таймлайну
DEFAULT_TIMELINE_BINS = 24

# Доля "топ-бинов" для оценки концентрации (например 0.2 = топ 20% бинов)
DEFAULT_TIMELINE_TOP_FRAC = 0.2


# -----------------------------
# Вспомогательные структуры
# -----------------------------

@dataclass(frozen=True)
class BestRule:
    profit_col: str
    profit_sum: float
    n_rows: int
    profit_mean: float
    rule_values: Dict[str, Tuple[Any, ...]]  # col -> tuple(values)
    rule_text: str

    # Битовые маски по категориям (чтобы быстро восстановить mask по df без df.isin)
    bit_masks: Dict[str, int]  # col -> int bitmask

    # Метрики "равномерности по таймлайну" (опционально)
    timeline_bins: Optional[int] = None
    timeline_coverage: Optional[float] = None          # доля бинов, где правило встречалось
    timeline_evenness: Optional[float] = None          # 0..1 (1 = равномерно, 0 = концентрация)
    timeline_top_share: Optional[float] = None         # доля |прибыль|, пришедшаяся на топ-bins
    timeline_pos_bins: Optional[int] = None            # сколько бинов дали плюс
    timeline_neg_bins: Optional[int] = None            # сколько бинов дали минус


class _Progress:
    """
    Единый прогресс-бар с ETA.
    Обновление идёт батчами, чтобы не замедлять перебор.
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

        # fallback без tqdm: печать раз в несколько секунд
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


# -----------------------------
# Сборка тензора сумм по категориям
# -----------------------------

def _factorize_columns(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in cols:
        codes, uniques = pd.factorize(df[c], sort=True)
        out[c] = {
            "codes": codes.astype(np.int64, copy=False),
            "uniques": uniques.to_list(),
            "k": int(len(uniques)),
        }
    return out


def _build_tensor_sums(
    codes_list: Sequence[np.ndarray],
    shape: Tuple[int, ...],
    profit: np.ndarray,
) -> np.ndarray:
    """
    Возвращает тензор сумм profit по всем комбинациям категорий.
    """
    flat_idx = np.ravel_multi_index(tuple(codes_list), dims=shape, mode="raise")
    size = int(np.prod(shape, dtype=np.int64))
    sums_flat = np.bincount(flat_idx, weights=profit, minlength=size).astype(np.float64, copy=False)
    return sums_flat.reshape(shape)


def _build_tensor_counts(
    codes_list: Sequence[np.ndarray],
    shape: Tuple[int, ...],
) -> np.ndarray:
    """
    Возвращает тензор количества строк по всем комбинациям категорий.
    """
    flat_idx = np.ravel_multi_index(tuple(codes_list), dims=shape, mode="raise")
    size = int(np.prod(shape, dtype=np.int64))
    cnt_flat = np.bincount(flat_idx, minlength=size).astype(np.int64, copy=False)
    return cnt_flat.reshape(shape)


# -----------------------------
# Решатели (max_var 1..3)
# -----------------------------

def _solve_1d_best(
    w: np.ndarray,  # shape (k,)
    c: np.ndarray,  # shape (k,)
    min_rows: int,
    progress: Optional[_Progress] = None,
) -> Tuple[int, float, int]:
    """
    Возвращает:
      - mask (битовая маска выбранных значений)
      - best_profit_sum
      - best_count
    Полный перебор по всем непустым подмножествам.
    """
    k = int(w.shape[0])
    best_mask = 0
    best_sum = -float("inf")
    best_cnt = 0

    w = w.astype(np.float64, copy=False)
    c = c.astype(np.int64, copy=False)

    for mask in range(1, (1 << k)):
        s = 0.0
        n = 0
        for i in range(k):
            if (mask >> i) & 1:
                s += float(w[i])
                n += int(c[i])

        if progress is not None:
            progress.add(1)

        if n < min_rows:
            continue
        if s > best_sum or (s == best_sum and n > best_cnt):
            best_sum = s
            best_cnt = n
            best_mask = mask

    return best_mask, float(best_sum), int(best_cnt)


def _solve_2d_best(
    w: np.ndarray,  # shape (k1, k2)
    c: np.ndarray,  # shape (k1, k2)
    min_rows: int,
    progress: Optional[_Progress] = None,
) -> Tuple[int, int, float, int]:
    """
    Точный поиск максимальной суммы по "прямоугольнику" подмножеств:
      выбираем подмножество строк и подмножество столбцов.
    """
    k1, k2 = int(w.shape[0]), int(w.shape[1])

    # Чтобы было быстрее — перебираем подмножества по меньшей оси.
    if k1 <= k2:
        small_axis = 0
        w_small = w
        c_small = c
    else:
        small_axis = 1
        w_small = w.T
        c_small = c.T
        k1, k2 = k2, k1  # теперь k1 = k_small, k2 = k_large

    k_small = k1
    k_large = k2

    base_w = w_small.astype(np.float64, copy=False)  # shape (k_small, k_large)
    base_c = c_small.astype(np.int64, copy=False)

    best_mask_small = 0
    best_mask_large = 0
    best_sum = -float("inf")
    best_cnt = 0

    sum_w = np.zeros((1 << k_small, k_large), dtype=np.float64)
    sum_c = np.zeros((1 << k_small, k_large), dtype=np.int64)

    for mask in range(1, (1 << k_small)):
        lsb = mask & -mask
        bit = (lsb.bit_length() - 1)
        prev = mask ^ lsb

        sum_w[mask] = sum_w[prev] + base_w[bit]
        sum_c[mask] = sum_c[prev] + base_c[bit]

    for mask_small in range(1, (1 << k_small)):
        vec_w = sum_w[mask_small]
        vec_c = sum_c[mask_small]

        local_best_mask_large = 0
        local_best_sum = -float("inf")
        local_best_cnt = 0

        for mask_large in range(1, (1 << k_large)):
            s = 0.0
            n = 0
            for j in range(k_large):
                if (mask_large >> j) & 1:
                    s += float(vec_w[j])
                    n += int(vec_c[j])

            if n < min_rows:
                continue
            if s > local_best_sum or (s == local_best_sum and n > local_best_cnt):
                local_best_sum = s
                local_best_cnt = n
                local_best_mask_large = mask_large

        if progress is not None:
            progress.add(1)

        if local_best_mask_large == 0:
            continue

        if local_best_sum > best_sum or (local_best_sum == best_sum and local_best_cnt > best_cnt):
            best_sum = local_best_sum
            best_cnt = local_best_cnt
            best_mask_small = mask_small
            best_mask_large = local_best_mask_large

    if small_axis == 0:
        return int(best_mask_small), int(best_mask_large), float(best_sum), int(best_cnt)
    else:
        return int(best_mask_large), int(best_mask_small), float(best_sum), int(best_cnt)


def _solve_3d_best(
    w: np.ndarray,  # shape (k1, k2, k3)
    c: np.ndarray,  # shape (k1, k2, k3)
    min_rows: int,
    progress: Optional[_Progress] = None,
) -> Tuple[int, int, int, float, int, Tuple[int, int, int]]:
    """
    Точный поиск по 3 столбцам.
    """
    ks = [int(w.shape[0]), int(w.shape[1]), int(w.shape[2])]
    perm = tuple(int(i) for i in np.argsort(ks))  # от меньшей уникальности к большей
    w_p = np.transpose(w, axes=perm).astype(np.float64, copy=False)
    c_p = np.transpose(c, axes=perm).astype(np.int64, copy=False)

    kA, kB, kC = int(w_p.shape[0]), int(w_p.shape[1]), int(w_p.shape[2])

    sumW_A = np.zeros((1 << kA, kB, kC), dtype=np.float64)
    sumC_A = np.zeros((1 << kA, kB, kC), dtype=np.int64)

    for maskA in range(1, (1 << kA)):
        lsb = maskA & -maskA
        bit = (lsb.bit_length() - 1)
        prev = maskA ^ lsb
        sumW_A[maskA] = sumW_A[prev] + w_p[bit]
        sumC_A[maskA] = sumC_A[prev] + c_p[bit]

    bestA = 0
    bestB = 0
    bestC = 0
    best_sum = -float("inf")
    best_cnt = 0

    for maskA in range(1, (1 << kA)):
        w2d = sumW_A[maskA]
        c2d = sumC_A[maskA]

        maskB, maskC, s, n = _solve_2d_best(w2d, c2d, min_rows=min_rows, progress=None)

        if progress is not None:
            progress.add(1)

        if maskB == 0 or maskC == 0:
            continue

        if s > best_sum or (s == best_sum and n > best_cnt):
            best_sum = s
            best_cnt = n
            bestA = maskA
            bestB = maskB
            bestC = maskC

    return int(bestA), int(bestB), int(bestC), float(best_sum), int(best_cnt), perm


# -----------------------------
# Форматирование правил
# -----------------------------

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


# -----------------------------
# Метрики "равномерности по таймлайну"
# -----------------------------

def _bitmask_to_allowed_array(bitmask: int, k: int) -> np.ndarray:
    # k маленький (<= max_unique), поэтому простая генерация ok
    arr = np.zeros(k, dtype=np.bool_)
    for i in range(k):
        if (bitmask >> i) & 1:
            arr[i] = True
    return arr


def _calc_timeline_metrics(
    *,
    fact: Dict[str, Dict[str, Any]],
    cols: Tuple[str, ...],
    bit_masks: Dict[str, int],
    profit: np.ndarray,
    bin_id: np.ndarray,
    n_bins: int,
    top_frac: float,
) -> Dict[str, Any]:
    """
    Считает:
      - coverage: доля бинов, где правило реально встречалось (есть строки)
      - evenness: 0..1 по распределению |прибыль по бинам|
      - top_share: какая доля |прибыль| пришлась на топ top_frac бинов
      - pos_bins / neg_bins: сколько бинов дали плюс/минус (с учётом наличия строк)
    """
    n = int(profit.shape[0])
    if n == 0 or n_bins <= 0:
        return {
            "coverage": None,
            "evenness": None,
            "top_share": None,
            "pos_bins": None,
            "neg_bins": None,
        }

    mask = np.ones(n, dtype=np.bool_)
    for c in cols:
        codes = fact[c]["codes"]
        k = int(fact[c]["k"])
        bm = int(bit_masks[c])
        allowed = _bitmask_to_allowed_array(bm, k)
        mask &= allowed[codes]

        # ранний выход
        if not mask.any():
            return {
                "coverage": 0.0,
                "evenness": 0.0,
                "top_share": 0.0,
                "pos_bins": 0,
                "neg_bins": 0,
            }

    b = bin_id[mask]
    p = profit[mask]

    cnt_per_bin = np.bincount(b, minlength=n_bins).astype(np.int64, copy=False)
    sum_per_bin = np.bincount(b, weights=p, minlength=n_bins).astype(np.float64, copy=False)

    has = cnt_per_bin > 0
    bins_hit = int(has.sum())
    coverage = float(bins_hit / n_bins) if n_bins > 0 else 0.0

    pos_bins = int(((sum_per_bin > 0) & has).sum())
    neg_bins = int(((sum_per_bin < 0) & has).sum())

    abs_sum = np.abs(sum_per_bin)
    total_abs = float(abs_sum.sum())
    if total_abs <= 0.0:
        return {
            "coverage": coverage,
            "evenness": 0.0,
            "top_share": 0.0,
            "pos_bins": pos_bins,
            "neg_bins": neg_bins,
        }

    # top_share: доля |прибыль| в топ X% бинов
    top_k = int(max(1, math.ceil(n_bins * float(top_frac))))
    top_share = float(np.sort(abs_sum)[::-1][:top_k].sum() / total_abs)

    # evenness: нормированная энтропия по |прибыль|
    nonzero = abs_sum[abs_sum > 0]
    if nonzero.size <= 1:
        evenness = 1.0
    else:
        probs = nonzero / nonzero.sum()
        ent = float(-(probs * np.log(probs)).sum())
        ent_max = float(np.log(nonzero.size))
        evenness = float(ent / ent_max) if ent_max > 0 else 0.0

    return {
        "coverage": coverage,
        "evenness": evenness,
        "top_share": top_share,
        "pos_bins": pos_bins,
        "neg_bins": neg_bins,
    }


# -----------------------------
# Основная функция
# -----------------------------

def find_best_profit_rules(
    csv_path: str,
    *,
    max_var: int = 2,
    profit_cols: Tuple[str, str] = DEFAULT_PROFIT_COLS,
    exclude_cols: Tuple[str, ...] = DEFAULT_EXCLUDE_COLS,
    max_unique: int = DEFAULT_MAX_UNIQUE,
    min_rows: int = DEFAULT_MIN_ROWS,
    show_progress: bool = True,
    print_results: bool = True,

    # NEW: по какому профиту сортировать вывод в конце
    # None -> по второму из profit_cols (как было раньше)
    sort_profit_col: Optional[str] = None,

    # Новое: включать/выключать оценку равномерности по таймлайну
    calc_timeline_uniformity: bool = False,
    timeline_bins: int = DEFAULT_TIMELINE_BINS,
    timeline_top_frac: float = DEFAULT_TIMELINE_TOP_FRAC,
    sort_by_tm_ms: bool = True,
) -> Dict[Tuple[str, ...], Dict[str, BestRule]]:
    """
    Ищет лучшие правила вида "col in (..)" для всех комбинаций столбцов размера max_var.

    Оптимизация делается по сумме профита (profit_sum) на выбранных строках.
    Для устойчивости можно задать min_rows, чтобы правило не выбиралось по слишком малому числу строк.
    """
    if not isinstance(max_var, int) or max_var < 1:
        raise ValueError("max_var must be an integer >= 1.")
    if max_var > 3:
        raise ValueError("This module implements exact search for max_var up to 3. Increase carefully if you extend it.")
    if not isinstance(timeline_bins, int) or timeline_bins < 2:
        raise ValueError("timeline_bins must be an integer >= 2.")
    if not (0.0 < float(timeline_top_frac) <= 1.0):
        raise ValueError("timeline_top_frac must be in (0, 1].")

    df = pd.read_csv(csv_path)

    # -----------------------------
    # NEW: фильтрация по периоду sv.START / sv.END (по колонке tm_ms)
    # Логика:
    #   - START включительно (оставляем всё, что не раньше START)
    #   - END не включительно (оставляем всё, что строго раньше END)
    # Если START/END не заданы (None) — фильтрации нет.
    # -----------------------------
    start_dt = getattr(sv, "START", None)
    end_dt = getattr(sv, "END", None)

    if (start_dt is not None) or (end_dt is not None):
        if "tm_ms" not in df.columns:
            raise ValueError("To filter by period you need 'tm_ms' column in CSV.")

        tm_ms = pd.to_numeric(df["tm_ms"], errors="coerce")
        if tm_ms.isna().all():
            raise ValueError("'tm_ms' column could not be parsed as numbers.")

        df = df.copy()
        df["tm_ms"] = tm_ms.astype(np.int64, copy=False)

        def _dt_to_ms(dt: datetime) -> int:
            # Если datetime без timezone — считаем, что это UTC.
            # Если timezone есть — приводим к UTC.
            if dt.tzinfo is None:
                dt_utc = dt.replace(tzinfo=timezone.utc)
            else:
                dt_utc = dt.astimezone(timezone.utc)

            # Переводим datetime в миллисекунды с начала эпохи Unix (UTC)
            return int(dt_utc.timestamp() * 1000)

        if start_dt is not None:
            start_ms = _dt_to_ms(start_dt)
            df = df[df["tm_ms"] >= start_ms]

        if end_dt is not None:
            end_ms = _dt_to_ms(end_dt)
            df = df[df["tm_ms"] < end_ms]

        df = df.reset_index(drop=True)

    # (опционально) сортируем по tm_ms, чтобы гарантировать хронологию
    if sort_by_tm_ms and "tm_ms" in df.columns:
        # mergesort: стабильная сортировка
        df = df.sort_values("tm_ms", kind="mergesort").reset_index(drop=True)


    for pc in profit_cols:
        if pc not in df.columns:
            raise ValueError(f"Profit column '{pc}' not found in CSV.")
    # NEW: выбираем, по какому profit сортировать финальный вывод
    sort_pc = sort_profit_col if sort_profit_col is not None else profit_cols[1]
    if sort_pc not in profit_cols:
        raise ValueError(f"sort_profit_col must be one of {profit_cols}, got: {sort_pc}")

    feature_cols = [c for c in df.columns if c not in profit_cols and c not in exclude_cols]
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions.")

    # Фильтруем по уникальности + сразу выкидываем константные столбцы (в т.ч. все нули)
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
        if nun <= max_unique:
            eligible_cols.append(c)
        else:
            skipped_cols.append((c, nun))

    if print_results:
        if skipped_constant:
            print(f"Skipped constant/all-zero columns: {len(skipped_constant)}")
            for c, v in skipped_constant:
                print(f"  - {c}: constant={v}")

        print("=" * 100)
        print(f"CSV: {csv_path}")
        print(f"Rows: {len(df):,}")
        print(f"Eligible columns (nunique <= {max_unique}): {len(eligible_cols)}")
        if skipped_cols:
            print(f"Skipped columns (too many unique values): {len(skipped_cols)}")
            for c, nun in skipped_cols:
                print(f"  - {c}: nunique={nun}")
        print(f"Search: max_var={max_var}, min_rows={min_rows}, profits={profit_cols}")
        if calc_timeline_uniformity:
            print(f"Timeline uniformity: enabled | bins={timeline_bins} | top_frac={timeline_top_frac}")
        else:
            print("Timeline uniformity: disabled")
        print("=" * 100)

    if len(eligible_cols) < max_var:
        raise ValueError(f"Not enough eligible columns for max_var={max_var}. Eligible: {len(eligible_cols)}")

    # Факторизация всех eligible столбцов один раз
    fact = _factorize_columns(df, eligible_cols)

    # Все комбинации столбцов размера max_var
    col_sets = list(itertools.combinations(eligible_cols, max_var))

    # Оценка "веса" для прогресса
    def estimate_weight(cols: Tuple[str, ...]) -> int:
        ks = [int(fact[c]["k"]) for c in cols]
        if max_var == 1:
            base = (1 << ks[0]) - 1
        elif max_var == 2:
            base = (1 << min(ks[0], ks[1])) - 1
        else:
            ks_sorted = sorted(ks)
            base = ((1 << ks_sorted[0]) - 1) * ((1 << min(ks_sorted[1], ks_sorted[2])) - 1)
        return int(base) * len(profit_cols)

    total_weight = sum(estimate_weight(cs) for cs in col_sets)
    progress = _Progress(total=total_weight, enabled=show_progress)

    results: Dict[Tuple[str, ...], Dict[str, BestRule]] = {}

    # numpy профиты
    profit_arrays = {pc: df[pc].to_numpy(dtype=np.float64, copy=False) for pc in profit_cols}

    # подготовка bin_id для таймлайна (если надо)
    if calc_timeline_uniformity:
        n_rows_total = int(len(df))
        idx = np.arange(n_rows_total, dtype=np.int64)
        # bin_id: равные по количеству строк сегменты
        bin_id = (idx * int(timeline_bins) // max(1, n_rows_total)).astype(np.int64, copy=False)
        # защита от попадания последней строки в bin == timeline_bins
        bin_id = np.minimum(bin_id, int(timeline_bins) - 1)
    else:
        bin_id = None  # type: ignore

    try:
        for cols in col_sets:
            codes_list = [fact[c]["codes"] for c in cols]
            uniques_list = [fact[c]["uniques"] for c in cols]
            shape = tuple(int(fact[c]["k"]) for c in cols)

            cnt_tensor = _build_tensor_counts(codes_list, shape)

            per_profit: Dict[str, BestRule] = {}

            for pc in profit_cols:
                w_tensor = _build_tensor_sums(codes_list, shape, profit_arrays[pc])

                if max_var == 1:
                    w1 = w_tensor.reshape(shape[0])
                    c1 = cnt_tensor.reshape(shape[0])
                    mask0, best_sum, best_cnt = _solve_1d_best(w1, c1, min_rows=min_rows, progress=progress)

                    vals0 = _mask_to_values(mask0, uniques_list[0])
                    rule_vals = {cols[0]: vals0}
                    rule_txt = _rule_text(rule_vals)
                    bit_masks = {cols[0]: int(mask0)}

                elif max_var == 2:
                    w2 = w_tensor.reshape(shape[0], shape[1])
                    c2 = cnt_tensor.reshape(shape[0], shape[1])
                    mask0, mask1, best_sum, best_cnt = _solve_2d_best(w2, c2, min_rows=min_rows, progress=progress)

                    vals0 = _mask_to_values(mask0, uniques_list[0])
                    vals1 = _mask_to_values(mask1, uniques_list[1])
                    rule_vals = {cols[0]: vals0, cols[1]: vals1}
                    rule_txt = _rule_text(rule_vals)
                    bit_masks = {cols[0]: int(mask0), cols[1]: int(mask1)}

                else:
                    w3 = w_tensor.reshape(shape[0], shape[1], shape[2])
                    c3 = cnt_tensor.reshape(shape[0], shape[1], shape[2])
                    mA, mB, mC, best_sum, best_cnt, perm = _solve_3d_best(w3, c3, min_rows=min_rows, progress=progress)

                    masks_perm = [0, 0, 0]
                    masks_perm[0] = mA
                    masks_perm[1] = mB
                    masks_perm[2] = mC

                    masks_orig = [0, 0, 0]
                    for new_axis, old_axis in enumerate(perm):
                        masks_orig[old_axis] = masks_perm[new_axis]

                    vals0 = _mask_to_values(masks_orig[0], uniques_list[0])
                    vals1 = _mask_to_values(masks_orig[1], uniques_list[1])
                    vals2 = _mask_to_values(masks_orig[2], uniques_list[2])
                    rule_vals = {cols[0]: vals0, cols[1]: vals1, cols[2]: vals2}
                    rule_txt = _rule_text(rule_vals)
                    bit_masks = {cols[0]: int(masks_orig[0]), cols[1]: int(masks_orig[1]), cols[2]: int(masks_orig[2])}

                mean = float(best_sum / best_cnt) if best_cnt > 0 else float("nan")

                per_profit[pc] = BestRule(
                    profit_col=pc,
                    profit_sum=float(best_sum),
                    n_rows=int(best_cnt),
                    profit_mean=mean,
                    rule_values=rule_vals,
                    rule_text=rule_txt,
                    bit_masks=bit_masks,
                    # timeline_* пока пустые, добавим ниже (опционально)
                )

            results[tuple(cols)] = per_profit

        # Второй этап: метрики по таймлайну (если включено)
        if calc_timeline_uniformity:
            # отдельный прогресс, потому что этот этап не учитывался в estimate_weight
            timeline_total = int(len(col_sets) * len(profit_cols))
            tl_progress = _Progress(total=timeline_total, enabled=show_progress, update_every=256)

            try:
                for cols in col_sets:
                    rr = results.get(tuple(cols))
                    if not rr:
                        continue

                    for pc in profit_cols:
                        br = rr.get(pc)
                        if br is None:
                            tl_progress.add(1)
                            continue

                        metrics = _calc_timeline_metrics(
                            fact=fact,
                            cols=tuple(cols),
                            bit_masks=br.bit_masks,
                            profit=profit_arrays[pc],
                            bin_id=bin_id,  # type: ignore
                            n_bins=int(timeline_bins),
                            top_frac=float(timeline_top_frac),
                        )

                        rr[pc] = BestRule(
                            profit_col=br.profit_col,
                            profit_sum=br.profit_sum,
                            n_rows=br.n_rows,
                            profit_mean=br.profit_mean,
                            rule_values=br.rule_values,
                            rule_text=br.rule_text,
                            bit_masks=br.bit_masks,

                            timeline_bins=int(timeline_bins),
                            timeline_coverage=metrics["coverage"],
                            timeline_evenness=metrics["evenness"],
                            timeline_top_share=metrics["top_share"],
                            timeline_pos_bins=metrics["pos_bins"],
                            timeline_neg_bins=metrics["neg_bins"],
                        )

                        tl_progress.add(1)
            finally:
                tl_progress.close()

        return results

    finally:
        progress.close()

        if print_results:
            print()
            print("=" * 100)
            print("RESULTS")
            print("=" * 100)

            # Печатаем, отсортировав по выбранной колонке sort_pc (по возрастанию profit_sum)
            def _sort_key(cs: Tuple[str, ...]) -> float:
                rr = results.get(tuple(cs))
                if not rr:
                    return float("inf")
                br = rr.get(sort_pc)
                if br is None:
                    return float("inf")
                return float(br.profit_sum)

            sorted_col_sets = sorted(col_sets, key=_sort_key, reverse=False)

            for cols in sorted_col_sets:
                rr = results.get(tuple(cols))
                if not rr:
                    continue

                print("-" * 100)
                print(f"COLUMNS: {', '.join(cols)}")

                for pc in profit_cols:
                    br = rr[pc]
                    print(f"  [{pc}] best profit_sum={br.profit_sum:.6f} | N={br.n_rows} | mean={br.profit_mean:.6f}")
                    print(f"       rule: {br.rule_text}")

                    if calc_timeline_uniformity and br.timeline_bins is not None:
                        cov = br.timeline_coverage if br.timeline_coverage is not None else float("nan")
                        eve = br.timeline_evenness if br.timeline_evenness is not None else float("nan")
                        top = br.timeline_top_share if br.timeline_top_share is not None else float("nan")
                        pb = br.timeline_pos_bins if br.timeline_pos_bins is not None else 0
                        nb = br.timeline_neg_bins if br.timeline_neg_bins is not None else 0

                        print(
                            f"       timeline: bins={br.timeline_bins} | coverage={cov:.3f} | evenness={eve:.3f} "
                            f"| top_share={top:.3f} | pos_bins={pb} | neg_bins={nb}"
                        )

            print("-" * 100)


# -----------------------------
# Пример использования
# -----------------------------

def example_run() -> None:
    res = find_best_profit_rules(
        csv_path=CSV_PATH,
        max_var=MAX_VAR,
        max_unique=MAX_UNIQUE,
        min_rows=10,
        show_progress=True,
        print_results=True,

        # включить/выключить метрики равномерности
        sort_profit_col="profit_2",
        calc_timeline_uniformity=True,
        timeline_bins=BINS,
        timeline_top_frac=0.2,
        sort_by_tm_ms=True,
    )
    # res — словарь со всеми лучшими правилами


if __name__ == "__main__":
    # По вашему правилу: без CLI/argparse.
    example_run()
    pass
