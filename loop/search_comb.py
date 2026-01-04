#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gating_combo_search.py

Поиск сильных комбинаций категориальных фич для отсечения сделок (гейтинг).

Идея:
- Берём сделки (каждая имеет profit и dict features).
- Для каждой реально встречающейся комбинации вида:
    (cls_1h=8) или (cls_1h=8 & cls_15m=3) или ... до размера K
  считаем, что будет если ВСЕ такие сделки "вырезать":
    - сколько убытков (плохого профита) исчезнет
    - сколько прибыли (хорошего профита) тоже исчезнет
    - чистый эффект = насколько изменится общий profit

Функция печатает топ-результаты и возвращает структуру данных с деталями.
CLI нет (по просьбе), всё управляется параметрами.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import time


# =============================================================================
# Progress bar (tqdm optional)
# =============================================================================

class _SimpleProgress:
    def __init__(self, total: int, desc: str = "Progress", update_every: int = 2000) -> None:
        self.total = max(1, int(total))
        self.desc = desc
        self.update_every = max(1, int(update_every))
        self.i = 0
        self.t0 = time.time()
        self._last_print = 0

    def update(self, n: int = 1) -> None:
        self.i += n
        if (self.i - self._last_print) >= self.update_every or self.i >= self.total:
            self._last_print = self.i
            frac = self.i / self.total
            pct = frac * 100.0
            elapsed = max(1e-9, time.time() - self.t0)
            speed = self.i / elapsed
            eta = (self.total - self.i) / max(1e-9, speed)
            bar_len = 22
            filled = int(bar_len * frac)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r{self.desc}: [{bar}] {pct:6.2f}% | {self.i}/{self.total} | "
                f"{speed:,.0f}/s | ETA {eta:,.1f}s",
                end="",
            )
            if self.i >= self.total:
                print()

    def close(self) -> None:
        if self.i < self.total:
            self.update(self.total - self.i)


def _iter_with_progress(items: Sequence[Any], desc: str) -> Iterable[Any]:
    """
    ВАЖНО: эта функция должна ВСЕГДА yield-ить элементы.
    Иначе, если внутри есть yield хотя бы в одной ветке,
    любая "return ..." превращается в досрочное завершение генератора.
    """
    total = len(items)

    # Пытаемся tqdm
    try:
        from tqdm import tqdm  # type: ignore
        for x in tqdm(items, total=total, desc=desc):
            yield x
        return
    except Exception:
        pass

    # Фолбек: простой прогресс печатью
    if total <= 0:
        return

    t0 = time.time()
    step = 2000  # как часто печатать прогресс
    for i, x in enumerate(items, 1):
        if i == 1 or i == total or (i % step == 0):
            elapsed = max(1e-9, time.time() - t0)
            speed = i / elapsed
            eta = (total - i) / max(1e-9, speed)
            pct = (i / total) * 100.0
            print(f"\r{desc}: {pct:6.2f}% | {i}/{total} | {speed:,.0f}/s | ETA {eta:,.1f}s", end="")
            if i == total:
                print()
        yield x


# =============================================================================
# Core structures
# =============================================================================

ComboKey = Tuple[Tuple[str, int], ...]  # (("cls_1h", 8), ("cls_15m", 3)) - всегда отсортировано по имени


@dataclass
class ComboAgg:
    count: int = 0
    sum_profit: float = 0.0
    sum_pos_profit: float = 0.0    # сумма только положительных profit
    sum_neg_profit: float = 0.0    # сумма только отрицательных profit (отрицательное число)
    pos_count: int = 0
    neg_count: int = 0

    def add(self, p: float) -> None:
        self.count += 1
        self.sum_profit += p
        if p >= 0:
            self.sum_pos_profit += p
            self.pos_count += 1
        else:
            self.sum_neg_profit += p
            self.neg_count += 1


@dataclass
class ComboRule:
    combo: Dict[str, int]

    # support
    count: int
    share_trades: float

    # group profit composition
    group_sum_profit: float
    group_mean_profit: float
    removed_good_profit: float
    removed_bad_profit_abs: float
    neg_trade_share: float

    # global impact if excluded
    total_profit_before: float
    total_profit_after: float
    profit_improvement: float

    # shares vs global totals
    share_of_all_bad_removed: float
    share_of_all_good_removed: float

    # ranking
    score: float


# =============================================================================
# Helpers
# =============================================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _combo_to_str(combo: Dict[str, int]) -> str:
    # стабильный порядок
    parts = [f"{k}={combo[k]}" for k in sorted(combo.keys())]
    return " & ".join(parts)


def _format_money(x: float) -> str:
    # без лишней “математики” в тексте: просто удобный формат
    return f"{x:,.4f}"


# =============================================================================
# Main function
# =============================================================================

def find_best_exclusion_combos(
    trades: List[Dict[str, Any]],
    *,
    max_combo_size: int = 2,
    min_combo_size: int = 1,
    combo_sizes: Optional[Sequence[int]] = None,
    min_support: int = 80,
    max_removed_trade_share: float = 0.35,
    min_profit_improvement: float = 0.0,
    good_profit_penalty: float = 0.15,
    stability_bonus: float = 0.05,
    top_n_print: int = 40,
    feature_whitelist: Optional[Sequence[str]] = None,
    signals: Optional[Sequence[int]] = None,
    profit_key: str = "profit",
    features_key: str = "features",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Ищет комбинации feature=value (размером min..K) которые стоит "банить" (вырезать),
    чтобы убрать максимум плохого профита и минимум хорошего.

    Что считается:
    - для каждой комбинации считаем сумму profit по сделкам, которые в неё попали.
    - если эту комбинацию "вырезать", общий профит станет:
        взять общий профит и отнять суммарный профит этой группы
      поэтому прирост равен:
        взять ноль и отнять суммарный профит группы
      (если группа убыточная, то прирост получается положительным)

    Рейтинг (score) по умолчанию:
    - берём чистый прирост
    - вычитаем штраф за вырезание хорошего профита (good_profit_penalty)
    - добавляем небольшой бонус за устойчивость по количеству (stability_bonus)

    Параметры:
      max_combo_size: максимальное число фич в комбинации
      min_combo_size: минимальное число фич в комбинации (ограничение снизу)
      combo_sizes: если задано, считаем только эти размеры (например [2])
      min_support: минимальное количество сделок в комбинации
      max_removed_trade_share: не предлагать комбо, которое вырежет слишком большой % сделок
      min_profit_improvement: минимальный чистый прирост профита, чтобы считать правило полезным
      good_profit_penalty: штраф в рейтинге за вырезание хорошего профита
      stability_bonus: бонус за размер выборки (чтобы маленькие случайные группы не лезли в топ)
      feature_whitelist: если задан, используем только эти feature-ключи
      signals: если задан, используем только сделки с trade['signal'] в этом списке
    """
    if not isinstance(trades, list):
        raise TypeError("trades must be a list of dicts")

    if min_combo_size < 1:
        raise ValueError("min_combo_size must be >= 1")

    if combo_sizes is None:
        if max_combo_size < 1:
            raise ValueError("max_combo_size must be >= 1")
        if min_combo_size > max_combo_size:
            raise ValueError("min_combo_size must be <= max_combo_size")
        sizes = list(range(min_combo_size, max_combo_size + 1))
    else:
        sizes = sorted({int(s) for s in combo_sizes if int(s) >= 1 and int(s) >= min_combo_size})
        if not sizes:
            raise ValueError("combo_sizes is empty or invalid (after applying min_combo_size)")

    # фильтруем сделки (по signal, если надо)
    filtered: List[Dict[str, Any]] = []
    if signals is None:
        filtered = trades
    else:
        sset = set(int(x) for x in signals)
        for t in trades:
            try:
                if int(t.get("signal", -999999)) in sset:
                    filtered.append(t)
            except Exception:
                continue

    n_total = len(filtered)
    if n_total == 0:
        return {
            "summary": {"trades_total": 0, "rules_found": 0},
            "rules": [],
        }

    # считаем общие суммы (для процентов)
    total_profit = 0.0
    total_good = 0.0
    total_bad_abs = 0.0  # сумма модулей отрицательных profit

    for t in filtered:
        p = _safe_float(t.get(profit_key, 0.0), 0.0)
        total_profit += p
        if p >= 0:
            total_good += p
        else:
            total_bad_abs += (-p)

    # whitelist
    whitelist = None
    if feature_whitelist is not None:
        whitelist = set(str(x) for x in feature_whitelist)

    # агрегаты по комбинациям
    agg: Dict[ComboKey, ComboAgg] = {}

    if verbose:
        print(f"[find_best_exclusion_combos] trades used: {n_total}")
        print(f"[find_best_exclusion_combos] combo sizes: {sizes} | min_support={min_support}")
        if signals is not None:
            print(f"[find_best_exclusion_combos] signals filter: {list(signals)}")
        if whitelist is not None:
            print(f"[find_best_exclusion_combos] feature_whitelist: {sorted(whitelist)}")
        print("[find_best_exclusion_combos] building aggregates...")

    it = _iter_with_progress(filtered, desc="Scanning trades")
    for t in it:
        p = _safe_float(t.get(profit_key, 0.0), 0.0)
        f = t.get(features_key, {}) or {}
        if not isinstance(f, dict):
            continue

        items: List[Tuple[str, int]] = []
        for k, v in f.items():
            if k is None:
                continue
            k_str = str(k)
            if whitelist is not None and k_str not in whitelist:
                continue
            try:
                iv = int(v)
            except Exception:
                continue
            items.append((k_str, iv))

        if not items:
            continue

        # чтобы ключи были стабильны: сортируем по имени фичи
        items.sort(key=lambda x: x[0])

        # генерим только заданные размеры
        m = len(items)
        for sz in sizes:
            if sz > m:
                continue
            for comb in combinations(items, sz):
                key: ComboKey = tuple(comb)
                a = agg.get(key)
                if a is None:
                    a = ComboAgg()
                    agg[key] = a
                a.add(p)

    if verbose:
        print(f"[find_best_exclusion_combos] unique combos seen: {len(agg):,}")
        print("[find_best_exclusion_combos] scoring candidates...")

    rules: List[ComboRule] = []

    # защита от деления на ноль в процентах
    denom_good = max(1e-12, total_good)
    denom_bad = max(1e-12, total_bad_abs)

    for key, a in agg.items():
        if a.count < min_support:
            continue

        share_trades = a.count / n_total
        if share_trades > max_removed_trade_share:
            continue

        group_sum = a.sum_profit
        # смысловой фильтр: правило "полезно", если при вырезании общий профит растёт
        profit_improvement = (0.0 - group_sum)
        if profit_improvement < min_profit_improvement:
            continue

        # “плохой профит” группы — это сумма модулей отрицательных сделок в группе
        removed_bad_abs = -a.sum_neg_profit  # a.sum_neg_profit <= 0
        removed_good = a.sum_pos_profit

        group_mean = group_sum / max(1, a.count)
        neg_trade_share = a.neg_count / max(1, a.count)

        total_after = total_profit - group_sum

        share_bad_removed = removed_bad_abs / denom_bad
        share_good_removed = removed_good / denom_good

        # рейтинг: чистый эффект минус штраф за вырезание хорошего + бонус за устойчивость
        # (стабильность: чем больше примеров, тем меньше риск случайности)
        stability = math.log1p(a.count)
        score = profit_improvement - (good_profit_penalty * removed_good) + (stability_bonus * stability * profit_improvement)

        combo_dict = {k: v for (k, v) in key}

        rules.append(
            ComboRule(
                combo=combo_dict,
                count=a.count,
                share_trades=share_trades,
                group_sum_profit=group_sum,
                group_mean_profit=group_mean,
                removed_good_profit=removed_good,
                removed_bad_profit_abs=removed_bad_abs,
                neg_trade_share=neg_trade_share,
                total_profit_before=total_profit,
                total_profit_after=total_after,
                profit_improvement=profit_improvement,
                share_of_all_bad_removed=share_bad_removed,
                share_of_all_good_removed=share_good_removed,
                score=score,
            )
        )

    # сортировка: сначала по score, потом по чистому улучшению, потом меньше вырезаем сделок, потом меньше вырезаем хорошего
    rules.sort(
        key=lambda r: (
            -r.score,
            -r.profit_improvement,
            r.share_trades,
            r.removed_good_profit,
        )
    )

    if verbose:
        print()
        print("=" * 110)
        print("TOP кандидаты для отсечения (ban rules)")
        print("- *Removed bad* = сколько убытков вырезаем (сумма модулей отрицательных profit в группе)")
        print("- *Removed good* = сколько прибыли случайно вырезаем (сумма положительных profit в группе)")
        print("- *Improvement* = насколько вырастет общий profit, если вырезать эту группу")
        print("=" * 110)

        show = rules[: max(0, int(top_n_print))]

        # ВАЖНО: combo НЕ обрезаем. Ширину колонки делаем по максимуму среди отображаемых строк.
        combo_col_width = 5
        if show:
            combo_col_width = max(combo_col_width, max(len(_combo_to_str(r.combo)) for r in show))

        header = (
            f"{'#':>3}  {'COMBO':<{combo_col_width}}  {'N':>6}  {'%N':>6}  "
            f"{'Removed bad':>13}  {'Removed good':>13}  {'Improvement':>13}  "
            f"{'%bad':>6}  {'%good':>6}  {'mean':>10}  {'neg%':>6}"
        )
        print(header)
        print("-" * len(header))

        for i, r in enumerate(show, 1):
            combo_str = _combo_to_str(r.combo)

            print(
                f"{i:>3}  {combo_str:<{combo_col_width}}  {r.count:>6}  {r.share_trades*100:>5.1f}%  "
                f"{_format_money(r.removed_bad_profit_abs):>13}  {_format_money(r.removed_good_profit):>13}  {_format_money(r.profit_improvement):>13}  "
                f"{r.share_of_all_bad_removed*100:>5.1f}%  {r.share_of_all_good_removed*100:>5.1f}%  "
                f"{_format_money(r.group_mean_profit):>10}  {r.neg_trade_share*100:>5.1f}%"
            )

        print("-" * len(header))
        print(f"Всего правил-кандидатов (после фильтров): {len(rules):,}")
        print(f"Общий profit до:  {_format_money(total_profit)}")
        if rules:
            # “что будет если применить ТОЛЬКО лучшее правило”
            best = rules[0]
            print(
                f"Если вырезать ТОП-1: общий profit станет {_format_money(best.total_profit_after)} "
                f"(прирост {_format_money(best.profit_improvement)})"
            )
        print("=" * 110)
        print()

    return {
        "summary": {
            "trades_total": n_total,
            "total_profit": total_profit,
            "total_good_profit": total_good,
            "total_bad_profit_abs": total_bad_abs,
            "combo_sizes": sizes,
            "min_support": min_support,
            "rules_found": len(rules),
            "params": {
                "max_removed_trade_share": max_removed_trade_share,
                "min_profit_improvement": min_profit_improvement,
                "good_profit_penalty": good_profit_penalty,
                "stability_bonus": stability_bonus,
            },
        },
        "rules": [asdict(r) for r in rules],
    }


# =============================================================================
# Optional: small convenience wrapper
# =============================================================================

def top_ban_combos(
    trades: List[Dict[str, Any]],
    *,
    k: int = 2,
    min_support: int = 80,
    top_n: int = 30,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Упрощённый вызов: только комбинации размера k и вернуть top_n.
    """
    out = find_best_exclusion_combos(
        trades,
        combo_sizes=[k],
        min_support=min_support,
        top_n_print=top_n,
        **kwargs,
    )
    return out["rules"][:top_n]
