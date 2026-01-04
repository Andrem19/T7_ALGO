from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Iterable
import random
import math
import time
import sys


RuleKey = Tuple  # ("pair", cat, int_value) | ("range", cat, kind, lo, hi, bin_start, bin_end) | ("and2", ...) | ("and3", ...)


# =============================================================================
# PASS RULES (single trade)
# =============================================================================
def pass_rules(
    rules: Dict[str, Any],
    features: Dict[str, Any],
) -> bool:
    """
    True  -> сделка ПРОХОДИТ фильтр (НЕ запрещена)
    False -> сделку надо ОТСЕЧЬ по rules

    rules:
      - banned_by_category: {cat: [int_values...]}
      - banned_float_ranges_by_category: {cat: [ {lo,hi,kind,...}, ... ]}
      - banned_combo_rules: [ {"size":2|3, "conds":{cat:int,...}}, ... ]

    Логика:
      1) combo-правила: если совпали ВСЕ условия => False
      2) int запреты: features[cat] == value => False
      3) float диапазоны: lo <= features[cat] <= hi => False
      иначе True
    """
    if not isinstance(features, dict) or not features:
        return True

    banned_by_category = rules.get("banned_by_category") or {}
    banned_float_ranges = rules.get("banned_float_ranges_by_category") or {}
    banned_combo_rules = rules.get("banned_combo_rules") or []

    # --- 1) combo rules ---
    if isinstance(banned_combo_rules, list) and banned_combo_rules:
        for cr in banned_combo_rules:
            if not isinstance(cr, dict):
                continue
            conds = cr.get("conds")
            if not isinstance(conds, dict) or not conds:
                continue

            ok_all = True
            for cat, need_v in conds.items():
                if cat not in features:
                    ok_all = False
                    break

                v = features.get(cat)

                iv: Optional[int] = None
                try:
                    if isinstance(v, bool):
                        iv = int(v)
                    elif isinstance(v, int):
                        iv = int(v)
                    elif isinstance(v, float) and math.isfinite(v) and abs(v - int(round(v))) < 1e-12:
                        iv = int(round(v))
                except Exception:
                    iv = None

                if iv is None:
                    ok_all = False
                    break

                try:
                    if iv != int(need_v):
                        ok_all = False
                        break
                except Exception:
                    ok_all = False
                    break

            if ok_all:
                return False

    # --- 2) int bans ---
    if isinstance(banned_by_category, dict) and banned_by_category:
        for cat, banned_vals in banned_by_category.items():
            if cat not in features:
                continue
            v = features.get(cat)

            iv: Optional[int] = None
            try:
                if isinstance(v, bool):
                    iv = int(v)
                elif isinstance(v, int):
                    iv = int(v)
                elif isinstance(v, float) and math.isfinite(v) and abs(v - int(round(v))) < 1e-12:
                    iv = int(round(v))
            except Exception:
                iv = None

            if iv is None:
                continue

            try:
                if iv in set(banned_vals):
                    return False
            except Exception:
                if isinstance(banned_vals, (list, tuple)) and iv in banned_vals:
                    return False

    # --- 3) float ranges ---
    if isinstance(banned_float_ranges, dict) and banned_float_ranges:
        for cat, ranges in banned_float_ranges.items():
            if cat not in features:
                continue

            v = features.get(cat)

            fv: Optional[float] = None
            try:
                if isinstance(v, bool):
                    fv = None
                elif isinstance(v, (int, float)):
                    fv = float(v)
                    if not math.isfinite(fv):
                        fv = None
            except Exception:
                fv = None

            if fv is None:
                continue

            if not isinstance(ranges, list):
                continue

            for rr in ranges:
                if not isinstance(rr, dict):
                    continue
                try:
                    lo = float(rr.get("lo"))
                    hi = float(rr.get("hi"))
                except Exception:
                    continue
                if lo > hi:
                    lo, hi = hi, lo
                if lo <= fv <= hi:
                    return False

    return True


# =============================================================================
# APPLY RULES (list of trades)
# =============================================================================
def apply_feature_exclusions(
    trades: List[Dict[str, Any]],
    *,
    banned_by_category: Optional[Dict[str, List[int]]] = None,
    banned_float_ranges_by_category: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    banned_combo_rules: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Возвращает (kept, removed)
    """
    rules = {
        "banned_by_category": banned_by_category or {},
        "banned_float_ranges_by_category": banned_float_ranges_by_category or {},
        "banned_combo_rules": banned_combo_rules or [],
    }

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for t in trades:
        feats = t.get("features") or {}
        if not isinstance(feats, dict):
            kept.append(t)
            continue

        if pass_rules(rules, feats):
            kept.append(t)
        else:
            removed.append(t)

    return kept, removed


# =============================================================================
# OPTIMISER
# =============================================================================
def optimise_feature_exclusions(
    trades: List[Dict[str, Any]],
    *,
    # ===== поиск =====
    max_rules: int = 30,
    restarts: int = 40,
    rcl_size: int = 8,
    worse_budget: int = 2,
    local_passes: int = 6,
    swap_attempts: int = 80,
    seed: int = 1,
    # ===== ограничения (общие, внутри каждого фолда) =====
    min_keep_trades: int = 0,
    min_keep_pos_profit: float = 0.0,
    min_keep_pos_profit_ratio: float = 0.0,
    # ===== float правила: один отрезок (середина или край) =====
    enable_float_ranges: bool = True,
    float_bins: int = 18,
    float_min_interval_trades: int = 200,          # важно: это требование на TRAIN части фолда
    float_min_interval_width_frac: float = 0.15,   # ширина отрезка относительно диапазона TRAIN
    float_allow_middle: bool = True,
    float_allow_edges: bool = True,
    float_middle_must_include_median: bool = True,
    float_interval_require_net_negative: bool = True,  # важно: net считается на TRAIN части
    float_max_intervals_per_feature: int = 1,       # максимум 1 отрезок на фичу
    # ===== авто-детект типов + ручные оверрайды =====
    force_float_features: Optional[List[str]] = None,
    force_int_features: Optional[List[str]] = None,
    min_feature_present: int = 30,                  # важно: это требование на TRAIN части фолда
    # ===== комбо-правила (2-3 int-фичи одновременно) =====
    enable_int_combo_rules: bool = True,
    combo_sizes: Tuple[int, ...] = (2, 3),
    combo2_top_per_feature: int = 14,
    combo3_top_per_feature: int = 8,
    combo2_min_trades: int = 25,                    # важно: это требование на TRAIN пересечении
    combo3_min_trades: int = 20,
    combo_require_net_negative: bool = True,        # важно: net считается на TRAIN пересечении
    combo2_max_candidates: int = 2500,
    combo3_max_candidates: int = 1200,
    # ===== переносимость на VAL (внутри каждого фолда) =====
    portability_weight: float = 2.0,                # в выборе правил val ценится сильнее train
    portability_min_val_trades: int = 80,           # если в val меньше — фолд пропускаем
    portability_min_val_improvement: float = 0.0,   # val после фильтра не должен быть хуже baseline
    portability_min_val_pos_profit_ratio: float = 0.0,  # на val сохранить долю валового плюса
    # ===== прогресс =====
    progress: bool = True,
    progress_every_restarts: int = 10,
    # ===== WALK-FORWARD (ЖЕЛЕЗНАЯ переносимость) =====
    walk_forward: bool = True,
    wf_mode: str = "expanding",     # "expanding" | "rolling"
    wf_train_min_trades: int = 450,
    wf_train_max_trades: Optional[int] = None,  # для rolling: размер окна train (если None => как получится)
    wf_val_trades: int = 150,
    wf_step_trades: Optional[int] = None,       # шаг сдвига; если None => равен wf_val_trades
    wf_max_folds: int = 12,
    # финальные правила: “ядро переносимости”
    wf_min_rule_support: float = 0.60,          # правило должно появляться минимум в такой доле фолдов
    wf_candidate_pool: int = 250,               # сколько кандидатов брать в жадный отбор
    wf_final_max_rules: int = 25,               # потолок на итоговое число правил
    wf_require_each_fold: bool = True,          # переносимость должна выполняться в каждом фолде
    # ===== debug =====
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Полноценный walk-forward:
      - датасет делится на фолды по времени (по индексу, данные уже в хронологическом порядке)
      - для каждого фолда:
          1) правила генерируются ТОЛЬКО по train-части
          2) оптимизация выбирает набор правил, при этом проверяет ограничения на val
      - затем собирается финальный “переносимый” набор правил:
          * берём правила, которые часто повторяются в фолдах
          * жадно добавляем их только если улучшают средний val результат
            и не нарушают требования переносимости
    """

    # ----------------------------
    # helpers
    # ----------------------------
    def _fmt_eta(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "?"
        s = int(seconds)
        m = s // 60
        h = m // 60
        s = s - m * 60
        m = m - h * 60
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def _to_float(x: Any) -> Optional[float]:
        try:
            if isinstance(x, bool):
                return None
            if isinstance(x, (int, float)):
                fv = float(x)
                if not math.isfinite(fv):
                    return None
                return fv
            return None
        except Exception:
            return None

    def _is_intlike_number(x: Any) -> bool:
        if isinstance(x, bool):
            return True
        if isinstance(x, int):
            return True
        if isinstance(x, float):
            if not math.isfinite(x):
                return False
            return abs(x - int(round(x))) < 1e-12
        return False

    def _to_int(x: Any) -> Optional[int]:
        try:
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, int):
                return int(x)
            if isinstance(x, float) and math.isfinite(x) and abs(x - int(round(x))) < 1e-12:
                return int(round(x))
            return None
        except Exception:
            return None

    def _intersect_sorted(a: List[int], b: List[int]) -> List[int]:
        i = 0
        j = 0
        out: List[int] = []
        while i < len(a) and j < len(b):
            ai = a[i]
            bj = b[j]
            if ai == bj:
                out.append(ai)
                i += 1
                j += 1
            elif ai < bj:
                i += 1
            else:
                j += 1
        return out

    # ----------------------------
    # fold optimiser (single fold)
    # ----------------------------
    def _optimise_one_fold(
        fold_trades: List[Dict[str, Any]],
        *,
        val_start_local: int,  # [val_start_local : end) is VAL
        fold_id: int,
    ) -> Dict[str, Any]:
        t_count = len(fold_trades)
        if t_count <= 0:
            return {}

        profits: List[float] = []
        feats_list: List[Dict[str, Any]] = []
        is_val: List[bool] = []

        for i, t in enumerate(fold_trades):
            profits.append(float(t.get("profit", 0.0)))
            feats = t.get("features") or {}
            feats_list.append(feats if isinstance(feats, dict) else {})
            is_val.append(i >= val_start_local)

        val_count = sum(1 for x in is_val if x)
        train_count = t_count - val_count

        if val_count < int(portability_min_val_trades):
            return {
                "skipped": True,
                "reason": f"val too small ({val_count} < {int(portability_min_val_trades)})",
                "metrics": {"fold_trades": t_count, "train_trades": train_count, "val_trades": val_count},
            }

        total_profit = sum(profits)
        total_pos_profit = sum(p for p in profits if p > 0.0)
        total_neg_abs = sum(-p for p in profits if p < 0.0)

        # split totals
        val_total_profit = 0.0
        val_total_pos_profit = 0.0
        val_total_neg_abs = 0.0
        train_total_profit = 0.0
        train_total_pos_profit = 0.0
        train_total_neg_abs = 0.0

        for i, p in enumerate(profits):
            if is_val[i]:
                val_total_profit += p
                if p > 0.0:
                    val_total_pos_profit += p
                elif p < 0.0:
                    val_total_neg_abs += -p
            else:
                train_total_profit += p
                if p > 0.0:
                    train_total_pos_profit += p
                elif p < 0.0:
                    train_total_neg_abs += -p

        min_keep_pos_by_ratio = total_pos_profit * float(min_keep_pos_profit_ratio)
        min_keep_val_pos_by_ratio = val_total_pos_profit * float(portability_min_val_pos_profit_ratio)

        # --------------- type detect ONLY on TRAIN ---------------
        force_float = set(force_float_features or [])
        force_int = set(force_int_features or [])

        present_train: Dict[str, int] = {}
        seen_non_int_float_train: Dict[str, bool] = {}

        for i, feats in enumerate(feats_list):
            if is_val[i]:
                continue
            for k, v in feats.items():
                cat = str(k)
                fv = _to_float(v)
                if fv is None:
                    continue
                present_train[cat] = present_train.get(cat, 0) + 1
                if not _is_intlike_number(fv):
                    seen_non_int_float_train[cat] = True

        float_feats_set = set()
        int_feats_set = set()

        for cat, cnt in present_train.items():
            if cnt < int(min_feature_present):
                continue

            if cat in force_float:
                float_feats_set.add(cat)
                continue
            if cat in force_int:
                int_feats_set.add(cat)
                continue

            if seen_non_int_float_train.get(cat, False):
                float_feats_set.add(cat)
            else:
                int_feats_set.add(cat)

        int_feats_set -= float_feats_set

        # --------------- build candidate rules from TRAIN only ---------------
        rule_id: Dict[RuleKey, int] = {}
        rule_keys: List[RuleKey] = []
        trades_of_rule: List[List[int]] = []
        float_range_meta: Dict[RuleKey, Dict[str, Any]] = {}

        def _add_rule(key: RuleKey, trade_indices: List[int]) -> int:
            rid = rule_id.get(key)
            if rid is not None:
                return rid
            rid = len(rule_keys)
            rule_id[key] = rid
            rule_keys.append(key)
            trades_of_rule.append(trade_indices)
            return rid

        # ---- int pair map across ALL fold trades, but eligibility is decided by TRAIN stats ----
        pair_all: Dict[Tuple[str, int], List[int]] = {}
        pair_train_cnt: Dict[Tuple[str, int], int] = {}
        pair_train_net: Dict[Tuple[str, int], float] = {}

        for ti, feats in enumerate(feats_list):
            for k, v in feats.items():
                cat = str(k)
                if cat not in int_feats_set:
                    continue
                iv = _to_int(v)
                if iv is None:
                    continue
                key = (cat, int(iv))
                pair_all.setdefault(key, []).append(ti)
                if not is_val[ti]:
                    pair_train_cnt[key] = pair_train_cnt.get(key, 0) + 1
                    pair_train_net[key] = pair_train_net.get(key, 0.0) + profits[ti]

        # atomic int rules
        for (cat, iv), idxs_all in pair_all.items():
            if pair_train_cnt.get((cat, iv), 0) < int(min_feature_present):
                continue
            _add_rule(("pair", cat, int(iv)), idxs_all)

        # ---- float rules: intervals are computed on TRAIN distribution and TRAIN profits only ----
        if enable_float_ranges and float_feats_set:
            bins = max(6, int(float_bins))

            for cat in sorted(float_feats_set):
                vt_train: List[Tuple[float, int]] = []
                for ti, feats in enumerate(feats_list):
                    if is_val[ti]:
                        continue
                    if cat not in feats:
                        continue
                    fv = _to_float(feats.get(cat))
                    if fv is None:
                        continue
                    vt_train.append((fv, ti))

                if len(vt_train) < max(50, int(float_min_interval_trades)):
                    continue

                vt_train.sort(key=lambda x: x[0])
                vals = [x[0] for x in vt_train]
                vmin = vals[0]
                vmax = vals[-1]
                span = vmax - vmin
                if span <= 1e-15:
                    continue

                b = min(bins, len(vt_train))
                edges: List[float] = [vmin]
                for bi in range(1, b):
                    pos = int(round((len(vals) - 1) * (bi / b)))
                    edges.append(vals[pos])
                edges.append(vmax)

                # bins stats on TRAIN only
                trades_in_bin_train: List[List[int]] = [[] for _ in range(b)]
                bin_net = [0.0] * b
                bin_pos = [0.0] * b
                bin_neg = [0.0] * b

                for j, (_fv, ti) in enumerate(vt_train):
                    bi = int((j * b) / len(vt_train))
                    if bi >= b:
                        bi = b - 1
                    trades_in_bin_train[bi].append(ti)
                    p = profits[ti]
                    bin_net[bi] += p
                    if p > 0.0:
                        bin_pos[bi] += p
                    elif p < 0.0:
                        bin_neg[bi] += -p

                median_bin = b // 2
                min_width = float_min_interval_width_frac * span

                def _eval_interval(kind: str, start: int, end: int) -> Optional[Tuple[float, RuleKey, List[int], Dict[str, Any]]]:
                    lo = edges[start]
                    hi = edges[end + 1]
                    width = hi - lo
                    if width < min_width:
                        return None

                    cnt_train = 0
                    net_train = 0.0
                    pos_train = 0.0
                    neg_train = 0.0

                    for bi in range(start, end + 1):
                        idxs_b = trades_in_bin_train[bi]
                        cnt_train += len(idxs_b)
                        net_train += bin_net[bi]
                        pos_train += bin_pos[bi]
                        neg_train += bin_neg[bi]

                    if cnt_train < int(float_min_interval_trades):
                        return None
                    if float_interval_require_net_negative and not (net_train < 0.0):
                        return None

                    # now build affected trades across ALL fold trades using these thresholds
                    affected_all: List[int] = []
                    for ti2, feats2 in enumerate(feats_list):
                        if cat not in feats2:
                            continue
                        fv2 = _to_float(feats2.get(cat))
                        if fv2 is None:
                            continue
                        if lo <= fv2 <= hi:
                            affected_all.append(ti2)

                    if not affected_all:
                        return None

                    key: RuleKey = ("range", cat, kind, float(lo), float(hi), int(start), int(end))
                    meta = {
                        "category": cat,
                        "kind": kind,
                        "lo": float(lo),
                        "hi": float(hi),
                        "bin_start": int(start),
                        "bin_end": int(end),
                        "train_trade_count": int(cnt_train),
                        "train_net_profit": float(net_train),
                        "train_pos_profit_sum": float(pos_train),
                        "train_neg_abs_sum": float(neg_train),
                        "width_frac_train": float(width / span) if span > 0 else 0.0,
                        "affected_total_trades": int(len(affected_all)),
                    }
                    score = -net_train  # score based ONLY on TRAIN
                    return (score, key, affected_all, meta)

                candidates: List[Tuple[float, RuleKey, List[int], Dict[str, Any]]] = []

                if float_allow_middle and b >= 6:
                    for start in range(1, b - 1):
                        for end in range(start, b - 2 + 1):
                            if float_middle_must_include_median and not (start <= median_bin <= end):
                                continue
                            cand = _eval_interval("middle", start, end)
                            if cand is not None:
                                candidates.append(cand)

                if float_allow_edges and b >= 4:
                    for end in range(0, b - 2 + 1):
                        cand = _eval_interval("left", 0, end)
                        if cand is not None:
                            candidates.append(cand)
                    for start in range(1, b - 1):
                        cand = _eval_interval("right", start, b - 1)
                        if cand is not None:
                            candidates.append(cand)

                if not candidates:
                    continue

                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[: max(1, int(float_max_intervals_per_feature))]

                for _score, key, affected_all, meta in candidates:
                    _add_rule(key, affected_all)
                    float_range_meta[key] = meta

        # ---- combo rules: selected ONLY by TRAIN intersection stats ----
        if enable_int_combo_rules and int_feats_set and combo_sizes:
            per_feature_bad: Dict[str, List[Tuple[str, int, List[int], float]]] = {}

            for (cat, iv), idxs_all in pair_all.items():
                cnt_train = pair_train_cnt.get((cat, iv), 0)
                if cnt_train < min(combo2_min_trades, combo3_min_trades):
                    continue
                net_train = pair_train_net.get((cat, iv), 0.0)
                if combo_require_net_negative and not (net_train < 0.0):
                    continue
                score = -net_train
                per_feature_bad.setdefault(cat, []).append((cat, iv, idxs_all, score))

            for cat in list(per_feature_bad.keys()):
                per_feature_bad[cat].sort(key=lambda x: x[3], reverse=True)
                keep_n = max(int(combo2_top_per_feature), int(combo3_top_per_feature))
                per_feature_bad[cat] = per_feature_bad[cat][:keep_n]

            cats = sorted([c for c in per_feature_bad.keys() if c in int_feats_set])

            def _train_stats_on_indices(idxs: List[int]) -> Tuple[int, float]:
                cnt = 0
                net = 0.0
                for ti in idxs:
                    if is_val[ti]:
                        continue
                    cnt += 1
                    net += profits[ti]
                return cnt, net

            if 2 in combo_sizes and len(cats) >= 2:
                pair_cands: List[Tuple[float, RuleKey, List[int]]] = []
                for i1 in range(len(cats)):
                    c1 = cats[i1]
                    list1 = per_feature_bad.get(c1, [])[: int(combo2_top_per_feature)]
                    if not list1:
                        continue
                    for i2 in range(i1 + 1, len(cats)):
                        c2 = cats[i2]
                        list2 = per_feature_bad.get(c2, [])[: int(combo2_top_per_feature)]
                        if not list2:
                            continue

                        for (_c1, v1, idxs1, _s1) in list1:
                            for (_c2, v2, idxs2, _s2) in list2:
                                inter = _intersect_sorted(idxs1, idxs2)
                                if not inter:
                                    continue
                                cnt_train, net_train = _train_stats_on_indices(inter)
                                if cnt_train < int(combo2_min_trades):
                                    continue
                                if combo_require_net_negative and not (net_train < 0.0):
                                    continue
                                score = -net_train
                                key: RuleKey = ("and2", c1, int(v1), c2, int(v2))
                                pair_cands.append((score, key, inter))

                if pair_cands:
                    pair_cands.sort(key=lambda x: x[0], reverse=True)
                    pair_cands = pair_cands[: max(0, int(combo2_max_candidates))]
                    for _score, key, inter in pair_cands:
                        _add_rule(key, inter)

            if 3 in combo_sizes and len(cats) >= 3:
                tri_cands: List[Tuple[float, RuleKey, List[int]]] = []
                top3_by_cat: Dict[str, List[Tuple[str, int, List[int]]]] = {}
                for c in cats:
                    top3_by_cat[c] = [(x[0], x[1], x[2]) for x in per_feature_bad.get(c, [])[: int(combo3_top_per_feature)]]

                for a in range(len(cats)):
                    c1 = cats[a]
                    l1 = top3_by_cat.get(c1, [])
                    if not l1:
                        continue
                    for b2 in range(a + 1, len(cats)):
                        c2 = cats[b2]
                        l2 = top3_by_cat.get(c2, [])
                        if not l2:
                            continue
                        for c3i in range(b2 + 1, len(cats)):
                            c3 = cats[c3i]
                            l3 = top3_by_cat.get(c3, [])
                            if not l3:
                                continue

                            for (_c1, v1, idxs1) in l1:
                                for (_c2, v2, idxs2) in l2:
                                    inter12 = _intersect_sorted(idxs1, idxs2)
                                    if not inter12:
                                        continue
                                    for (_c3, v3, idxs3) in l3:
                                        inter = _intersect_sorted(inter12, idxs3)
                                        if not inter:
                                            continue
                                        cnt_train, net_train = _train_stats_on_indices(inter)
                                        if cnt_train < int(combo3_min_trades):
                                            continue
                                        if combo_require_net_negative and not (net_train < 0.0):
                                            continue
                                        score = -net_train
                                        key: RuleKey = ("and3", c1, int(v1), c2, int(v2), c3, int(v3))
                                        tri_cands.append((score, key, inter))

                if tri_cands:
                    tri_cands.sort(key=lambda x: x[0], reverse=True)
                    tri_cands = tri_cands[: max(0, int(combo3_max_candidates))]
                    for _score, key, inter in tri_cands:
                        _add_rule(key, inter)

        r_count = len(rule_keys)
        if r_count == 0:
            return {
                "skipped": False,
                "banned_by_category": {},
                "banned_float_ranges_by_category": {},
                "banned_combo_rules": [],
                "metrics": {
                    "fold_trades": t_count,
                    "train_trades": train_count,
                    "val_trades": val_count,
                    "kept_trades": t_count,
                    "kept_profit": total_profit,
                    "val_kept_profit": val_total_profit,
                    "val_profit_improvement": 0.0,
                    "val_pos_profit_kept_ratio": 1.0 if val_total_pos_profit > 0 else 1.0,
                    "portability_used": True,
                    "rules_total": 0,
                },
            }

        # ----------------------------
        # State for GRASP
        # ----------------------------
        w = float(portability_weight)
        if not math.isfinite(w) or w < 1.0:
            w = 1.0

        class _State:
            __slots__ = (
                "banned",
                "banned_count",
                "counter",
                "kept",
                "kept_profit",
                "kept_count",
                "kept_pos_profit",
                "kept_neg_abs",
                "removed_pos_profit",
                "removed_neg_abs",
                "removed_count",
                # val
                "kept_profit_val",
                "kept_count_val",
                "kept_pos_profit_val",
                "kept_neg_abs_val",
                "removed_pos_profit_val",
                "removed_neg_abs_val",
                "removed_count_val",
            )

            def __init__(self) -> None:
                self.banned = [False] * r_count
                self.banned_count = 0
                self.counter = [0] * t_count
                self.kept = [True] * t_count

                self.kept_profit = total_profit
                self.kept_count = t_count
                self.kept_pos_profit = total_pos_profit
                self.kept_neg_abs = total_neg_abs

                self.removed_pos_profit = 0.0
                self.removed_neg_abs = 0.0
                self.removed_count = 0

                self.kept_profit_val = val_total_profit
                self.kept_count_val = val_count
                self.kept_pos_profit_val = val_total_pos_profit
                self.kept_neg_abs_val = val_total_neg_abs

                self.removed_pos_profit_val = 0.0
                self.removed_neg_abs_val = 0.0
                self.removed_count_val = 0

            def score(self) -> float:
                # nonval + w * val
                nonval_profit = self.kept_profit - self.kept_profit_val
                return nonval_profit + w * self.kept_profit_val

            def _can_keep_constraints(
                self,
                delta_count_total: int,
                delta_pos_profit_total: float,
                delta_profit_val: float,
                delta_pos_profit_val: float,
            ) -> bool:
                new_kept_count = self.kept_count + delta_count_total
                if new_kept_count < min_keep_trades:
                    return False

                new_kept_pos = self.kept_pos_profit + delta_pos_profit_total
                if new_kept_pos < min_keep_pos_profit:
                    return False
                if new_kept_pos < min_keep_pos_by_ratio:
                    return False

                # переносимость на val
                new_val_profit = self.kept_profit_val + delta_profit_val
                if new_val_profit < (val_total_profit + float(portability_min_val_improvement)):
                    return False

                new_val_pos = self.kept_pos_profit_val + delta_pos_profit_val
                if new_val_pos < min_keep_val_pos_by_ratio:
                    return False

                return True

            def effect_ban(self, rid: int) -> Tuple[float, int, float, float, float]:
                """
                dp_obj, dc_total, dpos_total, dp_val, dpos_val
                dp_val/dpos_val: изменение kept_*_val (новое минус старое)
                dp_obj: изменение score-части (nonval + w*val)
                """
                if self.banned[rid]:
                    return (0.0, 0, 0.0, 0.0, 0.0)

                dp_nonval = 0.0
                dpos_nonval = 0.0
                dc_nonval = 0

                dp_val = 0.0
                dpos_val = 0.0
                dc_val = 0

                for ti in trades_of_rule[rid]:
                    if self.counter[ti] != 0:
                        continue
                    p = profits[ti]
                    if is_val[ti]:
                        dp_val -= p
                        dc_val -= 1
                        if p > 0.0:
                            dpos_val -= p
                    else:
                        dp_nonval -= p
                        dc_nonval -= 1
                        if p > 0.0:
                            dpos_nonval -= p

                dc_total = dc_nonval + dc_val
                dpos_total = dpos_nonval + dpos_val
                dp_obj = dp_nonval + w * dp_val
                return (dp_obj, dc_total, dpos_total, dp_val, dpos_val)

            def effect_unban(self, rid: int) -> Tuple[float, int, float, float, float]:
                if not self.banned[rid]:
                    return (0.0, 0, 0.0, 0.0, 0.0)

                dp_nonval = 0.0
                dpos_nonval = 0.0
                dc_nonval = 0

                dp_val = 0.0
                dpos_val = 0.0
                dc_val = 0

                for ti in trades_of_rule[rid]:
                    if self.counter[ti] != 1:
                        continue
                    p = profits[ti]
                    if is_val[ti]:
                        dp_val += p
                        dc_val += 1
                        if p > 0.0:
                            dpos_val += p
                    else:
                        dp_nonval += p
                        dc_nonval += 1
                        if p > 0.0:
                            dpos_nonval += p

                dc_total = dc_nonval + dc_val
                dpos_total = dpos_nonval + dpos_val
                dp_obj = dp_nonval + w * dp_val
                return (dp_obj, dc_total, dpos_total, dp_val, dpos_val)

            def ban(self, rid: int) -> None:
                if self.banned[rid]:
                    return
                self.banned[rid] = True
                self.banned_count += 1

                for ti in trades_of_rule[rid]:
                    c = self.counter[ti]
                    self.counter[ti] = c + 1
                    if c != 0:
                        continue

                    self.kept[ti] = False
                    p = profits[ti]

                    self.kept_profit -= p
                    self.kept_count -= 1
                    self.removed_count += 1

                    if p > 0.0:
                        self.kept_pos_profit -= p
                        self.removed_pos_profit += p
                    elif p < 0.0:
                        na = -p
                        self.kept_neg_abs -= na
                        self.removed_neg_abs += na

                    if is_val[ti]:
                        self.kept_profit_val -= p
                        self.kept_count_val -= 1
                        self.removed_count_val += 1

                        if p > 0.0:
                            self.kept_pos_profit_val -= p
                            self.removed_pos_profit_val += p
                        elif p < 0.0:
                            na = -p
                            self.kept_neg_abs_val -= na
                            self.removed_neg_abs_val += na

            def unban(self, rid: int) -> None:
                if not self.banned[rid]:
                    return
                self.banned[rid] = False
                self.banned_count -= 1

                for ti in trades_of_rule[rid]:
                    c = self.counter[ti]
                    self.counter[ti] = c - 1
                    if c != 1:
                        continue

                    self.kept[ti] = True
                    p = profits[ti]

                    self.kept_profit += p
                    self.kept_count += 1
                    self.removed_count -= 1

                    if p > 0.0:
                        self.kept_pos_profit += p
                        self.removed_pos_profit -= p
                    elif p < 0.0:
                        na = -p
                        self.kept_neg_abs += na
                        self.removed_neg_abs -= na

                    if is_val[ti]:
                        self.kept_profit_val += p
                        self.kept_count_val += 1
                        self.removed_count_val -= 1

                        if p > 0.0:
                            self.kept_pos_profit_val += p
                            self.removed_pos_profit_val -= p
                        elif p < 0.0:
                            na = -p
                            self.kept_neg_abs_val += na
                            self.removed_neg_abs_val -= na

            def banned_rids(self) -> List[int]:
                return [i for i, b in enumerate(self.banned) if b]

        def _build_state_from_banned(banned_rids: List[int]) -> _State:
            st = _State()
            for rid in banned_rids:
                st.ban(rid)
            return st

        # ----------------------------
        # GRASP + local improve
        # ----------------------------
        def _construct(rng: random.Random) -> List[int]:
            st = _State()
            best_score = st.score()
            best_banned = st.banned_rids()
            worse_left = int(worse_budget)

            for _ in range(max_rules):
                cands: List[Tuple[float, int, int, float, float, float]] = []
                for rid in range(r_count):
                    if st.banned[rid]:
                        continue
                    dp_obj, dc_total, dpos_total, dp_val, dpos_val = st.effect_ban(rid)
                    if dc_total == 0:
                        continue
                    if not st._can_keep_constraints(dc_total, dpos_total, dp_val, dpos_val):
                        continue
                    cands.append((dp_obj, rid, dc_total, dpos_total, dp_val, dpos_val))

                if not cands:
                    break

                cands.sort(key=lambda x: x[0], reverse=True)
                best_dp = cands[0][0]

                if best_dp > 0.0:
                    pos_only = [c for c in cands if c[0] > 0.0]
                    top = pos_only[: max(1, min(rcl_size, len(pos_only)))]
                else:
                    if worse_left <= 0:
                        break
                    top = cands[: max(1, min(rcl_size, len(cands)))]
                    worse_left -= 1

                if len(top) == 1:
                    pick = top[0]
                else:
                    dps = [t[0] for t in top]
                    min_dp = min(dps)
                    weights = [(dp - min_dp) + 1e-6 for dp in dps]
                    pick = rng.choices(top, weights=weights, k=1)[0]

                _dp_obj, rid, _dc_total, _dpos_total, _dp_val, _dpos_val = pick
                st.ban(rid)

                sc = st.score()
                if sc > best_score:
                    best_score = sc
                    best_banned = st.banned_rids()

            return best_banned

        def _best_add(st: _State) -> Optional[int]:
            best_rid = None
            best_dp = 0.0
            for rid in range(r_count):
                if st.banned[rid]:
                    continue
                dp_obj, dc_total, dpos_total, dp_val, dpos_val = st.effect_ban(rid)
                if dc_total == 0:
                    continue
                if not st._can_keep_constraints(dc_total, dpos_total, dp_val, dpos_val):
                    continue
                if dp_obj > best_dp:
                    best_dp = dp_obj
                    best_rid = rid
            return best_rid

        def _best_remove(st: _State) -> Optional[int]:
            best_rid = None
            best_dp = 0.0
            for rid in range(r_count):
                if not st.banned[rid]:
                    continue
                dp_obj, _dc_total, _dpos_total, _dp_val, _dpos_val = st.effect_unban(rid)
                if dp_obj > best_dp:
                    best_dp = dp_obj
                    best_rid = rid
            return best_rid

        def _local_improve(banned_start: List[int], rng: random.Random) -> List[int]:
            st = _build_state_from_banned(banned_start)
            best_score = st.score()
            best_banned = st.banned_rids()

            for _ in range(local_passes):
                improved = False

                while st.banned_count < max_rules:
                    rid = _best_add(st)
                    if rid is None:
                        break
                    st.ban(rid)
                    improved = True
                    sc = st.score()
                    if sc > best_score:
                        best_score = sc
                        best_banned = st.banned_rids()

                while True:
                    rid = _best_remove(st)
                    if rid is None:
                        break
                    dp_obj, _dc_total, _dpos_total, _dp_val, _dpos_val = st.effect_unban(rid)
                    if dp_obj <= 0.0:
                        break
                    st.unban(rid)
                    improved = True
                    sc = st.score()
                    if sc > best_score:
                        best_score = sc
                        best_banned = st.banned_rids()

                if improved:
                    continue

                banned_rids = st.banned_rids()
                if not banned_rids:
                    break

                for _ in range(swap_attempts):
                    rid_remove = rng.choice(banned_rids)
                    st.unban(rid_remove)

                    if st.banned_count < max_rules:
                        rid_add = _best_add(st)
                        if rid_add is not None:
                            st.ban(rid_add)
                            improved = True
                            sc = st.score()
                            if sc > best_score:
                                best_score = sc
                                best_banned = st.banned_rids()
                            break

                    st.ban(rid_remove)

                if not improved:
                    break

            return best_banned

        # ----------------------------
        # restarts with progress
        # ----------------------------
        best_global_score = float("-inf")
        best_global_banned: List[int] = []
        best_global_val_impr = float("-inf")
        best_global_val_kept = float("-inf")

        rr = max(1, int(restarts))
        pe = max(1, int(progress_every_restarts))
        per_restart_times: List[float] = []

        for r in range(rr):
            t0 = time.time()

            rng = random.Random(seed + 10007 * (fold_id + 1) + 1000003 * r)
            b0 = _construct(rng)
            b1 = _local_improve(b0, rng)
            st_final = _build_state_from_banned(b1)

            ok = True
            if st_final.kept_count < min_keep_trades:
                ok = False
            if st_final.kept_pos_profit < min_keep_pos_profit:
                ok = False
            if st_final.kept_pos_profit < min_keep_pos_by_ratio:
                ok = False

            # portability checks on VAL
            if st_final.kept_profit_val < (val_total_profit + float(portability_min_val_improvement)):
                ok = False
            if st_final.kept_pos_profit_val < min_keep_val_pos_by_ratio:
                ok = False

            sc = st_final.score()
            val_impr = st_final.kept_profit_val - val_total_profit

            if ok and sc > best_global_score:
                best_global_score = sc
                best_global_banned = b1
                best_global_val_impr = val_impr
                best_global_val_kept = st_final.kept_profit_val

            dt = time.time() - t0
            per_restart_times.append(dt)

            if progress and ((r + 1) % pe == 0 or (r + 1) == rr):
                done = r + 1
                avg = sum(per_restart_times) / len(per_restart_times)
                eta = _fmt_eta((rr - done) * avg)
                print(
                    f"  [fold {fold_id}] restarts {done}/{rr} ({done/rr*100:5.1f}%) "
                    f"best_val_impr={best_global_val_impr:.2f} best_val_kept={best_global_val_kept:.2f} "
                    f"avg={avg:.2f}s ETA {eta}",
                    file=sys.stdout,
                )

        # ----------------------------
        # decode best rules
        # ----------------------------
        st = _build_state_from_banned(best_global_banned)
        banned_rules = [rule_keys[rid] for rid in best_global_banned]

        banned_by_category: Dict[str, List[int]] = {}
        banned_float_ranges_by_category: Dict[str, List[Dict[str, Any]]] = {}
        banned_combo_rules: List[Dict[str, Any]] = []

        for rk in banned_rules:
            if rk[0] == "pair":
                _, cat, iv = rk
                banned_by_category.setdefault(cat, []).append(int(iv))
            elif rk[0] == "range":
                _, cat, kind, lo, hi, bs, be = rk
                meta = float_range_meta.get(rk, {})
                banned_float_ranges_by_category.setdefault(cat, []).append({
                    "kind": kind,
                    "lo": float(lo),
                    "hi": float(hi),
                    "bin_start": int(bs),
                    "bin_end": int(be),
                    "train_trade_count": int(meta.get("train_trade_count", 0)),
                    "train_net_profit": float(meta.get("train_net_profit", 0.0)),
                    "train_pos_profit_sum": float(meta.get("train_pos_profit_sum", 0.0)),
                    "train_neg_abs_sum": float(meta.get("train_neg_abs_sum", 0.0)),
                    "width_frac_train": float(meta.get("width_frac_train", 0.0)),
                    "affected_total_trades": int(meta.get("affected_total_trades", 0)),
                })
            elif rk[0] == "and2":
                _, c1, v1, c2, v2 = rk
                banned_combo_rules.append({"size": 2, "conds": {str(c1): int(v1), str(c2): int(v2)}})
            elif rk[0] == "and3":
                _, c1, v1, c2, v2, c3, v3 = rk
                banned_combo_rules.append({"size": 3, "conds": {str(c1): int(v1), str(c2): int(v2), str(c3): int(v3)}})

        for cat in list(banned_by_category.keys()):
            banned_by_category[cat] = sorted(set(banned_by_category[cat]))

        # metrics
        val_impr = st.kept_profit_val - val_total_profit
        val_pos_kept_ratio = (st.kept_pos_profit_val / val_total_pos_profit) if val_total_pos_profit > 0 else 1.0

        metrics = {
            "fold_trades": t_count,
            "train_trades": train_count,
            "val_trades": val_count,

            "total_profit": total_profit,
            "kept_profit": st.kept_profit,
            "profit_improvement": st.kept_profit - total_profit,

            "val_total_profit": val_total_profit,
            "val_kept_profit": st.kept_profit_val,
            "val_profit_improvement": val_impr,
            "val_pos_profit_kept_ratio": val_pos_kept_ratio,

            "kept_trades": st.kept_count,
            "removed_trades": st.removed_count,

            "banned_rule_count": len(best_global_banned),
            "rules_total": r_count,
            "portability_used": True,
        }

        out_fold = {
            "skipped": False,
            "banned_by_category": banned_by_category,
            "banned_float_ranges_by_category": banned_float_ranges_by_category,
            "banned_combo_rules": banned_combo_rules,
            "metrics": metrics,
        }

        if debug:
            out_fold["debug"] = {
                "int_features": sorted(list(int_feats_set)),
                "float_features": sorted(list(float_feats_set)),
                "present_train_top": dict(sorted(present_train.items(), key=lambda x: -x[1])[:40]),
            }

        return out_fold

    # ----------------------------
    # walk-forward fold building
    # ----------------------------
    n = len(trades)
    if n == 0:
        return {"banned_by_category": {}, "banned_float_ranges_by_category": {}, "banned_combo_rules": [], "metrics": {}}

    if not walk_forward:
        # fallback: один фолд (train = первые 70%, val = последние 30%)
        train_frac = 0.70
        val_start = int(round(train_frac * n))
        if val_start < 1:
            val_start = 1
        if val_start > n - 1:
            val_start = n - 1
        fold_trades = trades[:]
        out1 = _optimise_one_fold(fold_trades, val_start_local=val_start, fold_id=0)
        if out1.get("skipped"):
            return {"banned_by_category": {}, "banned_float_ranges_by_category": {}, "banned_combo_rules": [], "metrics": out1.get("metrics", {})}
        return out1

    step = int(wf_step_trades) if wf_step_trades is not None else int(wf_val_trades)
    if step <= 0:
        step = int(wf_val_trades)
    if step <= 0:
        step = 50

    wf_val_tr = max(10, int(wf_val_trades))
    wf_train_min = max(50, int(wf_train_min_trades))

    folds: List[Dict[str, Any]] = []
    fold_meta: List[Dict[str, int]] = []

    # expanding/rolling windows by index count (data already chronological)
    train_end = wf_train_min
    fold_id = 0
    while fold_id < int(wf_max_folds):
        val_start = train_end
        val_end = val_start + wf_val_tr
        if val_end > n:
            break

        if wf_mode == "rolling":
            train_max = wf_train_max_trades
            if train_max is None or int(train_max) <= 0:
                # rolling but without explicit size: still cut start to keep reasonable
                train_start = max(0, val_start - wf_train_min)
            else:
                train_start = max(0, val_start - int(train_max))
        else:
            train_start = 0

        fold_start = train_start
        fold_end = val_end
        val_start_local = val_start - fold_start

        fold_meta.append({
            "fold_id": fold_id,
            "fold_start": fold_start,
            "fold_end": fold_end,
            "train_start": train_start,
            "train_end": val_start,
            "val_start": val_start,
            "val_end": val_end,
        })

        # advance
        train_end += step
        fold_id += 1

    if not fold_meta:
        # слишком мало данных для wf — fallback на holdout
        train_frac = 0.70
        val_start = int(round(train_frac * n))
        val_start = max(1, min(val_start, n - 1))
        out1 = _optimise_one_fold(trades[:], val_start_local=val_start, fold_id=0)
        if out1.get("skipped"):
            return {"banned_by_category": {}, "banned_float_ranges_by_category": {}, "banned_combo_rules": [], "metrics": out1.get("metrics", {})}
        return out1

    # ----------------------------
    # run folds
    # ----------------------------
    fold_start_time = time.time()
    fold_times: List[float] = []

    for i, fm in enumerate(fold_meta):
        t0 = time.time()

        if progress:
            done = i
            total = len(fold_meta)
            if done > 0 and fold_times:
                avg = sum(fold_times) / len(fold_times)
                eta = _fmt_eta((total - done) * avg)
            else:
                eta = "?"
            print(
                f"[WF] fold {i+1}/{len(fold_meta)} | "
                f"train {fm['train_start']}..{fm['train_end']} "
                f"val {fm['val_start']}..{fm['val_end']} | ETA {eta}",
                file=sys.stdout,
            )

        fold_trades = trades[fm["fold_start"]:fm["fold_end"]]
        out_fold = _optimise_one_fold(
            fold_trades,
            val_start_local=(fm["val_start"] - fm["fold_start"]),
            fold_id=i,
        )

        dt = time.time() - t0
        fold_times.append(dt)

        out_fold["fold"] = {
            "fold_id": fm["fold_id"],
            "fold_start": fm["fold_start"],
            "fold_end": fm["fold_end"],
            "train_start": fm["train_start"],
            "train_end": fm["train_end"],
            "val_start": fm["val_start"],
            "val_end": fm["val_end"],
        }
        folds.append(out_fold)

    # ----------------------------
    # build portable "core" rules from folds
    # ----------------------------
    valid_folds = [f for f in folds if not f.get("skipped")]
    if not valid_folds:
        return {
            "banned_by_category": {},
            "banned_float_ranges_by_category": {},
            "banned_combo_rules": [],
            "metrics": {"walk_forward": True, "folds_total": len(folds), "folds_used": 0},
            "walk_forward": {"folds": folds},
        }

    # signatures for stability counting
    def _sig_pair(cat: str, iv: int) -> Tuple:
        return ("pair", str(cat), int(iv))

    def _sig_combo(conds: Dict[str, int]) -> Tuple:
        items = tuple(sorted((str(k), int(v)) for k, v in conds.items()))
        return ("combo", items)

    def _sig_range(cat: str, kind: str, lo: float, hi: float) -> Tuple:
        # округление нужно, чтобы одинаковые “почти” совпали
        return ("range", str(cat), str(kind), round(float(lo), 4), round(float(hi), 4))

    sig_count: Dict[Tuple, int] = {}
    sig_payloads: Dict[Tuple, List[Dict[str, Any]]] = {}

    for f in valid_folds:
        bbc = f.get("banned_by_category") or {}
        bfr = f.get("banned_float_ranges_by_category") or {}
        bcr = f.get("banned_combo_rules") or []

        seen_in_fold = set()

        for cat, vals in bbc.items():
            if not isinstance(vals, list):
                continue
            for iv in vals:
                try:
                    sig = _sig_pair(cat, int(iv))
                except Exception:
                    continue
                seen_in_fold.add(sig)
                sig_payloads.setdefault(sig, []).append({"cat": str(cat), "iv": int(iv)})

        for cat, ranges in bfr.items():
            if not isinstance(ranges, list):
                continue
            for rr in ranges:
                if not isinstance(rr, dict):
                    continue
                try:
                    sig = _sig_range(cat, rr.get("kind", "range"), float(rr.get("lo")), float(rr.get("hi")))
                except Exception:
                    continue
                seen_in_fold.add(sig)
                sig_payloads.setdefault(sig, []).append({"cat": str(cat), "rr": rr})

        for cr in bcr:
            if not isinstance(cr, dict):
                continue
            conds = cr.get("conds")
            if not isinstance(conds, dict) or not conds:
                continue
            sig = _sig_combo(conds)
            seen_in_fold.add(sig)
            sig_payloads.setdefault(sig, []).append({"conds": {str(k): int(v) for k, v in conds.items()}})

        for sig in seen_in_fold:
            sig_count[sig] = sig_count.get(sig, 0) + 1

    folds_used = len(valid_folds)
    support_thr = float(wf_min_rule_support)
    if not math.isfinite(support_thr):
        support_thr = 0.60
    if support_thr < 0.05:
        support_thr = 0.05
    if support_thr > 1.0:
        support_thr = 1.0

    # candidates by support
    cand_sigs = []
    for sig, cnt in sig_count.items():
        support = cnt / folds_used
        if support + 1e-12 < support_thr:
            continue
        cand_sigs.append((support, cnt, sig))

    cand_sigs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    cand_sigs = cand_sigs[: max(1, int(wf_candidate_pool))]

    # evaluation of a ruleset on WF val parts (strict)
    def _eval_rules_on_wf_val(ruleset: Dict[str, Any]) -> Dict[str, Any]:
        val_kept_profit_sum = 0.0
        val_total_profit_sum = 0.0
        val_kept_trades_sum = 0
        val_total_trades_sum = 0

        val_pos_total_sum = 0.0
        val_pos_kept_sum = 0.0

        per_fold_ok = True
        per_fold_details = []

        for f in valid_folds:
            fm = f["fold"]
            val_start = fm["val_start"]
            val_end = fm["val_end"]

            # глобальные индексы val, и берём features/profit прямо из исходного trades
            kept_profit = 0.0
            total_profit = 0.0
            kept_tr = 0
            total_tr = 0

            pos_total = 0.0
            pos_kept = 0.0

            for gi in range(val_start, val_end):
                t = trades[gi]
                p = float(t.get("profit", 0.0))
                feats = t.get("features") or {}
                if not isinstance(feats, dict):
                    feats = {}

                total_profit += p
                total_tr += 1
                if p > 0.0:
                    pos_total += p

                if pass_rules(ruleset, feats):
                    kept_profit += p
                    kept_tr += 1
                    if p > 0.0:
                        pos_kept += p

            val_impr = kept_profit - total_profit
            pos_ratio = (pos_kept / pos_total) if pos_total > 0 else 1.0

            # strict portability checks on each fold
            fold_ok = True
            if kept_profit < (total_profit + float(portability_min_val_improvement)):
                fold_ok = False
            if pos_kept < (pos_total * float(portability_min_val_pos_profit_ratio)):
                fold_ok = False

            if wf_require_each_fold and not fold_ok:
                per_fold_ok = False

            per_fold_details.append({
                "val_trades": total_tr,
                "val_total_profit": total_profit,
                "val_kept_profit": kept_profit,
                "val_profit_improvement": val_impr,
                "val_pos_profit_kept_ratio": pos_ratio,
                "ok": fold_ok,
            })

            val_kept_profit_sum += kept_profit
            val_total_profit_sum += total_profit
            val_kept_trades_sum += kept_tr
            val_total_trades_sum += total_tr
            val_pos_total_sum += pos_total
            val_pos_kept_sum += pos_kept

        mean_val_profit_impr = (val_kept_profit_sum - val_total_profit_sum) / float(folds_used)
        mean_val_kept_trades = val_kept_trades_sum / float(folds_used)
        mean_val_total_trades = val_total_trades_sum / float(folds_used)
        mean_val_pos_ratio = (val_pos_kept_sum / val_pos_total_sum) if val_pos_total_sum > 0 else 1.0

        return {
            "ok": per_fold_ok,
            "mean_val_profit_improvement": mean_val_profit_impr,
            "mean_val_kept_trades": mean_val_kept_trades,
            "mean_val_total_trades": mean_val_total_trades,
            "mean_val_pos_profit_kept_ratio": mean_val_pos_ratio,
            "per_fold": per_fold_details,
        }

    def _ruleset_empty() -> Dict[str, Any]:
        return {"banned_by_category": {}, "banned_float_ranges_by_category": {}, "banned_combo_rules": []}

    def _ruleset_add_sig(ruleset: Dict[str, Any], sig: Tuple) -> None:
        if sig[0] == "pair":
            _, cat, iv = sig
            ruleset["banned_by_category"].setdefault(cat, [])
            if iv not in ruleset["banned_by_category"][cat]:
                ruleset["banned_by_category"][cat].append(iv)
        elif sig[0] == "combo":
            _, items = sig
            conds = {k: int(v) for (k, v) in items}
            # avoid duplicates
            for cr in ruleset["banned_combo_rules"]:
                if cr.get("conds") == conds:
                    return
            ruleset["banned_combo_rules"].append({"size": len(conds), "conds": conds})
        elif sig[0] == "range":
            _, cat, kind, lo, hi = sig
            rr = {"kind": kind, "lo": float(lo), "hi": float(hi)}
            ruleset["banned_float_ranges_by_category"].setdefault(cat, [])
            # avoid duplicates
            for ex in ruleset["banned_float_ranges_by_category"][cat]:
                if abs(float(ex.get("lo", 0.0)) - float(lo)) < 1e-12 and abs(float(ex.get("hi", 0.0)) - float(hi)) < 1e-12:
                    return
            ruleset["banned_float_ranges_by_category"][cat].append(rr)

    # greedy selection for portability
    base_rules = _ruleset_empty()
    base_eval = _eval_rules_on_wf_val(base_rules)

    best_rules = _ruleset_empty()
    best_eval = base_eval

    # Try to build portable ruleset
    chosen = 0
    for idx, (support, cnt, sig) in enumerate(cand_sigs):
        if chosen >= int(wf_final_max_rules):
            break

        trial = {
            "banned_by_category": {k: list(v) for k, v in best_rules["banned_by_category"].items()},
            "banned_float_ranges_by_category": {k: list(v) for k, v in best_rules["banned_float_ranges_by_category"].items()},
            "banned_combo_rules": list(best_rules["banned_combo_rules"]),
        }
        _ruleset_add_sig(trial, sig)

        ev = _eval_rules_on_wf_val(trial)
        if not ev["ok"]:
            continue

        if ev["mean_val_profit_improvement"] > best_eval["mean_val_profit_improvement"]:
            best_rules = trial
            best_eval = ev
            chosen += 1

    # sort int lists
    for cat in list(best_rules["banned_by_category"].keys()):
        best_rules["banned_by_category"][cat] = sorted(set(best_rules["banned_by_category"][cat]))

    # final metrics on full dataset (sanity)
    kept_all, removed_all = apply_feature_exclusions(
        trades,
        banned_by_category=best_rules["banned_by_category"],
        banned_float_ranges_by_category=best_rules["banned_float_ranges_by_category"],
        banned_combo_rules=best_rules["banned_combo_rules"],
    )
    full_total_profit = sum(float(t.get("profit", 0.0)) for t in trades)
    full_kept_profit = sum(float(t.get("profit", 0.0)) for t in kept_all)

    # ----------------------------
    # NEW: compact rules for copy/paste
    # ----------------------------
    def _norm_int_list(xs: Any) -> List[int]:
        outv: List[int] = []
        if not isinstance(xs, (list, tuple, set)):
            return outv
        for x in xs:
            try:
                outv.append(int(x))
            except Exception:
                pass
        return sorted(set(outv))

    # 1) compact banned_by_category
    _bbc_compact: Dict[str, List[int]] = {}
    for cat, vals in (best_rules["banned_by_category"] or {}).items():
        vv = _norm_int_list(vals)
        if vv:
            _bbc_compact[str(cat)] = vv

    # 2) compact float ranges: ONLY kind/lo/hi (pass_rules этого достаточно)
    _bfr_compact: Dict[str, List[Dict[str, Any]]] = {}
    for cat, ranges in (best_rules["banned_float_ranges_by_category"] or {}).items():
        if not isinstance(ranges, list):
            continue
        rr2: List[Dict[str, Any]] = []
        for rr in ranges:
            if not isinstance(rr, dict):
                continue
            try:
                lo = float(rr.get("lo"))
                hi = float(rr.get("hi"))
            except Exception:
                continue
            if lo > hi:
                lo, hi = hi, lo
            rr2.append({"kind": str(rr.get("kind", "range")), "lo": lo, "hi": hi})
        if rr2:
            _bfr_compact[str(cat)] = rr2

    # 3) compact combo rules
    _combo_compact: List[Dict[str, Any]] = []
    for cr in (best_rules["banned_combo_rules"] or []):
        if not isinstance(cr, dict):
            continue
        conds = cr.get("conds")
        if not isinstance(conds, dict) or not conds:
            continue
        conds2: Dict[str, int] = {}
        ok = True
        for k, v in conds.items():
            try:
                conds2[str(k)] = int(v)
            except Exception:
                ok = False
                break
        if not ok or not conds2:
            continue
        try:
            size = int(cr.get("size", len(conds2)))
        except Exception:
            size = len(conds2)
        _combo_compact.append({"size": size, "conds": conds2})

    _combo_compact.sort(key=lambda x: (x["size"], sorted(x["conds"].items())))

    rules_compact = {
        "banned_by_category": _bbc_compact,
        "banned_float_ranges_by_category": _bfr_compact,
        "banned_combo_rules": _combo_compact,
    }
    rules_compact_str = "sv.rules_1_1 = " + repr(rules_compact)

    # ----------------------------
    # your original out + add compact
    # ----------------------------
    out = {
        "banned_by_category": best_rules["banned_by_category"],
        "banned_float_ranges_by_category": best_rules["banned_float_ranges_by_category"],
        "banned_combo_rules": best_rules["banned_combo_rules"],
        "metrics": {
            "walk_forward": True,
            "folds_total": len(folds),
            "folds_used": folds_used,
            "full_total_trades": n,
            "full_kept_trades": len(kept_all),
            "full_removed_trades": len(removed_all),
            "full_total_profit": full_total_profit,
            "full_kept_profit": full_kept_profit,
            "full_profit_improvement": full_kept_profit - full_total_profit,
            "wf_mean_val_profit_improvement": best_eval["mean_val_profit_improvement"],
            "wf_mean_val_pos_profit_kept_ratio": best_eval["mean_val_pos_profit_kept_ratio"],
            "wf_mean_val_kept_trades": best_eval["mean_val_kept_trades"],
            "wf_mean_val_total_trades": best_eval["mean_val_total_trades"],
            "final_rule_count": (
                sum(len(v) for v in best_rules["banned_by_category"].values())
                + sum(len(v) for v in best_rules["banned_float_ranges_by_category"].values())
                + len(best_rules["banned_combo_rules"])
            ),
        },
        "walk_forward": {
            "folds": folds,
            "base_eval": base_eval,
            "final_eval": best_eval,
            "support_threshold": support_thr,
            "candidates_considered": len(cand_sigs),
            "wf_mode": wf_mode,
            "wf_train_min_trades": wf_train_min,
            "wf_val_trades": wf_val_tr,
            "wf_step_trades": step,
        },
    }

    # NEW keys:
    out["rules_compact"] = rules_compact
    out["rules_compact_str"] = rules_compact_str

    return out

