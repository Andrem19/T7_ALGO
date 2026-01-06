from __future__ import annotations

import shared_vars as sv
from itertools import combinations
from datetime import datetime
import helpers.util as util
import numpy as np
from loop.engine_1 import engine_1
from markov.hmm import infer_regime_latest
from helpers.util import find_candle_index
from typing import List, Sequence, Tuple, Optional, Dict, Any, Callable, Iterable


def _all_non_empty_subsets(values: Sequence[int]) -> List[Tuple[int, ...]]:
    """
    Все непустые подмножества.
    (0,1,2) -> (0), (1), (2), (0,1), (0,2), (1,2), (0,1,2)
    """
    vals = tuple(values)
    out: List[Tuple[int, ...]] = []
    for r in range(1, len(vals) + 1):
        out.extend(combinations(vals, r))
    return out


def build_all_combos(
    ch_1d_values: Optional[Sequence[int]] = (0, 1, 2),
    reg_3_values: Optional[Sequence[int]] = (0, 1, 2),
    dow_values: Optional[Sequence[int]] = (0, 1, 2, 3, 4, 5, 6),
) -> List[Dict[str, Any]]:
    """
    Простая функция без yield и без классов.

    Логика:
      - Если параметр передан НЕ None -> переменная участвует в переборе.
      - Если параметр None -> переменная исключена (её нет в комбинациях).

    Возвращает список комбинаций, где каждая комбинация — dict:
      {
        "ch_1d": (..),   # если участвует
        "reg_3": (..),   # если участвует
        "dow":   (..),   # если участвует
        "text": "ch_1d in (...) and dow in (...)"
      }
    """
    use_ch = ch_1d_values is not None
    use_rg = reg_3_values is not None
    use_dw = dow_values is not None

    if not (use_ch or use_rg or use_dw):
        raise ValueError("Нужно передать хотя бы одну переменную (не None).")

    # Готовим наборы (подмножества) только для тех переменных, которые участвуют
    if use_ch:
        ch_sets = _all_non_empty_subsets(ch_1d_values)
    if use_rg:
        rg_sets = _all_non_empty_subsets(reg_3_values)
    if use_dw:
        dw_sets = _all_non_empty_subsets(dow_values)

    res: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Обычные вложенные циклы: 3 переменные
    # -------------------------------------------------------------------------
    if use_ch and use_rg and use_dw:
        for ch in ch_sets:
            for rg in rg_sets:
                for dw in dw_sets:
                    text = f"ch_1d in {ch} and reg_3 in {rg} and dow in {dw}"
                    res.append({"ch_1d": ch, "reg_3": rg, "dow": dw, "text": text})
        return res

    # -------------------------------------------------------------------------
    # Обычные вложенные циклы: 2 переменные
    # -------------------------------------------------------------------------
    if use_ch and use_rg and not use_dw:
        for ch in ch_sets:
            for rg in rg_sets:
                text = f"ch_1d in {ch} and reg_3 in {rg}"
                res.append({"ch_1d": ch, "reg_3": rg, "text": text})
        return res

    if use_ch and use_dw and not use_rg:
        for ch in ch_sets:
            for dw in dw_sets:
                text = f"ch_1d in {ch} and dow in {dw}"
                res.append({"ch_1d": ch, "dow": dw, "text": text})
        return res

    if use_rg and use_dw and not use_ch:
        for rg in rg_sets:
            for dw in dw_sets:
                text = f"reg_3 in {rg} and dow in {dw}"
                res.append({"reg_3": rg, "dow": dw, "text": text})
        return res

    # -------------------------------------------------------------------------
    # Обычные циклы: 1 переменная
    # -------------------------------------------------------------------------
    if use_ch and not use_rg and not use_dw:
        for ch in ch_sets:
            text = f"ch_1d in {ch}"
            res.append({"ch_1d": ch, "text": text})
        return res

    if use_rg and not use_ch and not use_dw:
        for rg in rg_sets:
            text = f"reg_3 in {rg}"
            res.append({"reg_3": rg, "text": text})
        return res

    if use_dw and not use_ch and not use_rg:
        for dw in dw_sets:
            text = f"dow in {dw}"
            res.append({"dow": dw, "text": text})
        return res

    # Теоретически сюда не дойдём
    raise RuntimeError("Неожиданная комбинация входных параметров.")

def best_result_by_profit(
    results: Dict[str, Dict[str, Any]],
    *,
    min_all: int = 0,
    extra_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    profit_key: str = "profit",
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Берёт словарь вида:
        results[text] = {"plus": ..., "minus": ..., "all": ..., "profit": ..., ...}

    Фильтрует по условиям:
      - all >= min_all
      - extra_filter(name, item) == True (если передан)

    Возвращает (ключ, внутренний_словарь) с максимальным profit.
    Если после фильтрации ничего не осталось — вернёт None.
    """
    best_name: Optional[str] = None
    best_item: Optional[Dict[str, Any]] = None
    best_profit: Optional[float] = None

    for name, item in results.items():
        if not isinstance(item, dict):
            continue

        # 1) обязательное поле profit
        profit = item.get(profit_key, None)
        if profit is None:
            continue
        try:
            profit_f = float(profit)
        except (TypeError, ValueError):
            continue

        # 2) базовое условие: all не меньше заданного
        all_cnt = item.get("all", 0)
        try:
            all_cnt_i = int(all_cnt)
        except (TypeError, ValueError):
            continue
        if all_cnt_i < min_all:
            continue

        # 3) дополнительные условия через функцию-предикат
        if extra_filter is not None and not extra_filter(name, item):
            continue

        # 4) выбираем максимум profit
        if best_profit is None or profit_f > best_profit:
            best_profit = profit_f
            best_name = name
            best_item = item

    if best_name is None or best_item is None:
        return None
    return best_name, best_item
# =============================================================================
# HOW TO USE
# =============================================================================
def make_my_filter(
    *,
    min_ch_1d_len: int = 1,
    dow_any: Optional[Iterable[int]] = None,          # например (1, 2)
    ch_1d_must_contain: Optional[Iterable[int]] = None # например (2,) или (0, 2)
):
    """
    Возвращает функцию-предикат my_filter(name, item) -> bool,
    где item["ch_1d"] ожидается как set/tuple/list (например (0, 2)).
    """

    dow_any_set = set(dow_any) if dow_any is not None else None
    must_contain_set = set(ch_1d_must_contain) if ch_1d_must_contain is not None else None

    def my_filter(_name: str, it: Dict[str, Any]) -> bool:
        # --- ch_1d: приводим к set ---
        ch = it.get("ch_1d", None)

        if ch is None:
            ch_set: set[int] = set()
        elif isinstance(ch, int):
            ch_set = {ch}
        else:
            try:
                ch_set = set(ch)
            except TypeError:
                return False  # если там вдруг неитерируемое/битое значение

        # 1) минимальное кол-во элементов в ch_1d
        if len(ch_set) < int(min_ch_1d_len):
            return False

        # 2) опционально: ch_1d должен содержать заданные значения
        if must_contain_set is not None and not must_contain_set.issubset(ch_set):
            return False

        # --- dow: приводим к set ---
        dow = it.get("dow", None)
        if dow is None:
            dow_set: set[int] = set()
        elif isinstance(dow, int):
            dow_set = {dow}
        else:
            try:
                dow_set = set(dow)
            except TypeError:
                return False

        # 3) опционально: dow должен пересекаться с dow_any
        if dow_any_set is not None and dow_set.isdisjoint(dow_any_set):
            return False

        return True

    return my_filter

def get_best_combination(i, signal, days, ch_1d, dow):
    combos2 = build_all_combos(
        ch_1d_values=(ch_1d,),
        reg_3_values=None,
        dow_values=(dow,)
    )

    results = {}
    res_ch_cache = {}
    for rules in combos2:
        res = test_loop(i, days, rules, signal, res_ch_cache)
        results[rules['text']] = {
            'all': res['all'],
            'profit': res['profit'],
            'ch_1d': rules['ch_1d'],
            'dow': rules['dow']
        }
        
    my_filter = make_my_filter(min_ch_1d_len=1, dow_any=None)#, ch_1d_must_contain=ch_must_contein)
    best = best_result_by_profit(results, min_all=0, extra_filter=None)
    return best
    
    
def test_loop(it, days, rules, signal, res_ch_cache):
    i = it-days*24
    
    ch_1d_rule = rules['ch_1d']
    dow_rule = rules['dow']
    positions_list = []
    
    while i < it-24:
        dt = datetime.fromtimestamp(sv.data_1h[i][0]/1000)
        dow_now = dt.weekday()
        hour = dt.hour
        if hour not in [6]:
            i+=1
            continue
        if dow_now not in dow_rule:
            i+=1
            continue

        res_dict = sv.ch_res.get(sv.data_1h[i][0], None)

        if res_dict is None:
            i+=1
            continue
        res_ch = res_dict['ch_1d']
        if res_ch not in ch_1d_rule:
            i+=1
            continue
        
        result_1 = {'profit': res_dict['prof_1']} if signal == 1 else {'profit': res_dict['prof_2']}
        # print(result_1)

        positions_list.append(result_1)
        
        i+=22
        

    res = {
        'profit': sum(p['profit'] for p in positions_list),
        'all': len(positions_list)
    }
    return res