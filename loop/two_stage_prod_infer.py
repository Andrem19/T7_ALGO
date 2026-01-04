#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
two_stage_prod_infer.py

Инференс для PROD/PRED bundle из step_f_quant_cb_2.py.

Интерфейс:
- load_prod_bundle(prod_or_pred_dir) -> ProdBundle
- predict_one(bundle, features_dict, trade_frac=...) -> dict

Никакого CLI/argparse.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool


# =============================================================================
# DATA
# =============================================================================

@dataclass
class ProdBundle:
    prod_dir: str
    meta_path: Optional[str]

    feature_cols: List[str]
    num_feature_cols: List[str]
    cat_feature_cols: List[str]
    cat_feature_indices: List[int]

    x_good: float
    x_bad: float

    fill_num: Dict[str, float]
    fill_cat: Dict[str, Any]

    # e.g. {"0.20": 1.23, "0.10": 2.34}
    score_thresholds_from_oof: Dict[str, float]
    default_trade_frac: float

    # "logit_diff" (последний лучший) или "p_diff" (старый)
    score_mode: str

    gate_good: CatBoostClassifier
    gate_bad: CatBoostClassifier
    dir_cls: CatBoostClassifier
    pnl_reg: CatBoostRegressor


# =============================================================================
# IO HELPERS
# =============================================================================

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_json_files(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if fn.lower().endswith(".json"):
            out.append(os.path.join(d, fn))
    out.sort()
    return out


def _maybe_enter_prod_dir(path: str) -> str:
    """
    Можно передать:
    - путь прямо на .../prod или .../pred
    - путь на run folder, внутри которого есть prod/ или pred/
    """
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "gate_good.cbm")):
        return path

    for sub in ("prod", "pred"):
        p2 = os.path.join(path, sub)
        if os.path.isdir(p2) and os.path.isfile(os.path.join(p2, "gate_good.cbm")):
            return p2

    return path


def _find_any_meta_json(prod_dir: str) -> Optional[str]:
    """
    Раньше было prod_meta.json, сейчас у тебя есть metadata.json.
    Поэтому: если в папке есть хотя бы один json — берём его (предпочитая известные имена).
    """
    preferred = [
        "prod_meta.json",
        "pred_meta.json",
        "metadata.json",
        "meta.json",
        "bundle.json",
        "prod_bundle.json",
        "summary.json",
    ]
    for name in preferred:
        p = os.path.join(prod_dir, name)
        if os.path.isfile(p):
            return p

    j = _list_json_files(prod_dir)
    if len(j) == 1:
        return j[0]
    if len(j) > 1:
        # если несколько — берём самый “похожий” по имени
        return j[0]
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = -1) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def _get_by_path(obj: Any, path: List[Union[str, int]]) -> Any:
    """
    Достаёт вложенное значение:
      ["a","b",0,"c"] означает obj["a"]["b"][0]["c"].
    Возвращает None если не получилось.
    """
    cur = obj
    try:
        for p in path:
            if isinstance(p, int):
                cur = cur[p]
            else:
                cur = cur[p]
        return cur
    except Exception:
        return None


def _first_not_none(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


# =============================================================================
# META EXTRACTION (ROBUST)
# =============================================================================

def _model_feature_names(m: Any) -> Optional[List[str]]:
    # разные версии catboost имеют разные API
    for attr in ("feature_names_",):
        try:
            v = getattr(m, attr, None)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return list(v)
        except Exception:
            pass

    for meth in ("get_feature_names",):
        try:
            fn = getattr(m, meth, None)
            if callable(fn):
                v = fn()
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    return list(v)
        except Exception:
            pass

    return None


def _model_cat_indices(m: Any) -> Optional[List[int]]:
    for meth in ("get_cat_feature_indices",):
        try:
            fn = getattr(m, meth, None)
            if callable(fn):
                v = fn()
                if isinstance(v, (list, tuple)):
                    vv = [int(x) for x in v]
                    return vv
        except Exception:
            pass
    return None


def _extract_feature_cols(meta: Any, gate_good_model: CatBoostClassifier) -> List[str]:
    """
    Пытаемся получить список колонок:
    1) из meta (много вариантов ключей/вложенностей)
    2) из модели CatBoost
    """
    candidates = [
        meta.get("feature_cols") if isinstance(meta, dict) else None,
        meta.get("features") if isinstance(meta, dict) else None,
        meta.get("feature_names") if isinstance(meta, dict) else None,
        meta.get("columns") if isinstance(meta, dict) else None,
        _get_by_path(meta, ["data", "feature_cols"]),
        _get_by_path(meta, ["data", "feature_names"]),
        _get_by_path(meta, ["train", "feature_cols"]),
        _get_by_path(meta, ["train", "feature_names"]),
        _get_by_path(meta, ["schema", "feature_cols"]),
        _get_by_path(meta, ["schema", "feature_names"]),
    ]
    v = _first_not_none(*candidates)
    if isinstance(v, (list, tuple)) and len(v) > 0:
        return [str(x) for x in v]

    # fallback: из модели
    v2 = _model_feature_names(gate_good_model)
    if v2:
        return [str(x) for x in v2]

    # финально: ошибка с диагностикой
    keys = list(meta.keys()) if isinstance(meta, dict) else []
    raise ValueError(
        "Не удалось извлечь feature_cols из meta и из модели. "
        f"meta keys: {keys}"
    )


def _extract_cat_info(
    meta: Any,
    feature_cols: List[str],
    gate_good_model: CatBoostClassifier,
) -> Tuple[List[int], List[str], List[str]]:
    """
    Возвращает (cat_indices, cat_cols, num_cols).
    Пытаемся:
    - cat indices из meta (много вариантов)
    - cat cols из meta
    - если есть cat cols — пересчитываем indices
    - если нет ничего — пробуем из модели
    """
    cat_idx_candidates = [
        meta.get("cat_feature_indices") if isinstance(meta, dict) else None,
        meta.get("cat_features_indices") if isinstance(meta, dict) else None,
        meta.get("cat_indices") if isinstance(meta, dict) else None,
        meta.get("cat_features") if isinstance(meta, dict) else None,  # бывает список индексов
        _get_by_path(meta, ["data", "cat_feature_indices"]),
        _get_by_path(meta, ["train", "cat_feature_indices"]),
        _get_by_path(meta, ["schema", "cat_feature_indices"]),
    ]
    cat_cols_candidates = [
        meta.get("cat_feature_cols") if isinstance(meta, dict) else None,
        meta.get("cat_cols") if isinstance(meta, dict) else None,
        _get_by_path(meta, ["data", "cat_feature_cols"]),
        _get_by_path(meta, ["train", "cat_feature_cols"]),
        _get_by_path(meta, ["schema", "cat_feature_cols"]),
    ]
    num_cols_candidates = [
        meta.get("num_feature_cols") if isinstance(meta, dict) else None,
        meta.get("num_cols") if isinstance(meta, dict) else None,
        _get_by_path(meta, ["data", "num_feature_cols"]),
        _get_by_path(meta, ["train", "num_feature_cols"]),
        _get_by_path(meta, ["schema", "num_feature_cols"]),
    ]

    cat_indices_raw = _first_not_none(*cat_idx_candidates)
    cat_cols_raw = _first_not_none(*cat_cols_candidates)
    num_cols_raw = _first_not_none(*num_cols_candidates)

    cat_cols: List[str] = []
    num_cols: List[str] = []
    cat_indices: List[int] = []

    if isinstance(cat_cols_raw, (list, tuple)) and len(cat_cols_raw) > 0:
        cat_cols = [str(x) for x in cat_cols_raw]

    if isinstance(num_cols_raw, (list, tuple)) and len(num_cols_raw) > 0:
        num_cols = [str(x) for x in num_cols_raw]

    if isinstance(cat_indices_raw, (list, tuple)) and len(cat_indices_raw) > 0:
        # cat_features иногда может быть список имён, проверим тип первого
        if all(isinstance(x, (int, np.integer)) for x in cat_indices_raw):
            cat_indices = [int(x) for x in cat_indices_raw]
        else:
            # возможно это список имён колонок
            try:
                tmp_cols = [str(x) for x in cat_indices_raw]
                cat_cols = tmp_cols
            except Exception:
                pass

    # если есть cat_cols, но нет индексов — пересчитываем
    if cat_cols and not cat_indices:
        idx_map = {c: i for i, c in enumerate(feature_cols)}
        cat_indices = [idx_map[c] for c in cat_cols if c in idx_map]

    # если нет ничего — пробуем из модели
    if not cat_indices:
        mi = _model_cat_indices(gate_good_model)
        if mi:
            cat_indices = mi

    # если cat_cols не заполнены, но есть indices — восстановим
    if not cat_cols and cat_indices:
        cat_cols = [feature_cols[i] for i in cat_indices if 0 <= i < len(feature_cols)]

    # если num_cols пусты — восстановим как complement
    if not num_cols:
        cat_set = set(cat_cols)
        num_cols = [c for c in feature_cols if c not in cat_set]

    # финальная защита: indices должны быть уникальны и отсортированы
    cat_indices = sorted(list({int(i) for i in cat_indices if 0 <= int(i) < len(feature_cols)}))
    cat_cols = [feature_cols[i] for i in cat_indices]
    cat_set = set(cat_cols)
    num_cols = [c for c in feature_cols if c not in cat_set]

    return cat_indices, cat_cols, num_cols


def _extract_thresholds(meta: Any) -> Dict[str, float]:
    if not isinstance(meta, dict):
        return {}
    thr = meta.get("score_thresholds_from_oof")
    if thr is None:
        thr = meta.get("score_thresholds") or meta.get("oof_score_thresholds") or {}
    if not isinstance(thr, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in thr.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _extract_fills(meta: Any) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if not isinstance(meta, dict):
        return {}, {}
    fn = meta.get("fill_num", {})
    fc = meta.get("fill_cat", {})
    fill_num: Dict[str, float] = {}
    fill_cat: Dict[str, Any] = {}
    if isinstance(fn, dict):
        for k, v in fn.items():
            fill_num[str(k)] = _safe_float(v, 0.0)
    if isinstance(fc, dict):
        for k, v in fc.items():
            fill_cat[str(k)] = v
    return fill_num, fill_cat


# =============================================================================
# LOAD
# =============================================================================

def load_prod_bundle(prod_or_pred_dir: str) -> ProdBundle:
    prod_dir = _maybe_enter_prod_dir(prod_or_pred_dir)

    def _must_file(name: str) -> str:
        p = os.path.join(prod_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Model not found: {p}")
        return p

    # модели грузим сразу — они помогут восстановить мету при необходимости
    gate_good = CatBoostClassifier()
    gate_good.load_model(_must_file("gate_good.cbm"))

    gate_bad = CatBoostClassifier()
    gate_bad.load_model(_must_file("gate_bad.cbm"))

    dir_cls = CatBoostClassifier()
    dir_cls.load_model(_must_file("dir_cls.cbm"))

    pnl_reg = CatBoostRegressor()
    pnl_reg.load_model(_must_file("pnl_reg.cbm"))

    meta_path = _find_any_meta_json(prod_dir)
    meta = None
    if meta_path and os.path.isfile(meta_path):
        meta = _read_json(meta_path)
    else:
        meta_path = None
        meta = {}

    # feature cols
    feature_cols = _extract_feature_cols(meta, gate_good)

    # cat + num
    cat_indices, cat_cols, num_cols = _extract_cat_info(meta, feature_cols, gate_good)

    # x_good/x_bad (не обязаны быть)
    x_good = _safe_float(_first_not_none(
        meta.get("x_good") if isinstance(meta, dict) else None,
        _get_by_path(meta, ["gate", "x_good"]),
        _get_by_path(meta, ["params", "x_good"]),
    ), float("nan"))

    x_bad = _safe_float(_first_not_none(
        meta.get("x_bad") if isinstance(meta, dict) else None,
        _get_by_path(meta, ["gate", "x_bad"]),
        _get_by_path(meta, ["params", "x_bad"]),
    ), float("nan"))

    # fills + thresholds
    fill_num, fill_cat = _extract_fills(meta)
    score_thresholds_from_oof = _extract_thresholds(meta)

    # default_trade_frac + score_mode
    default_trade_frac = _safe_float(
        meta.get("default_trade_frac", meta.get("policy_trade_frac", 0.20)) if isinstance(meta, dict) else 0.20,
        0.20,
    )
    score_mode = str(meta.get("score_mode", "") if isinstance(meta, dict) else "").strip().lower()
    if not score_mode:
        score_mode = "logit_diff"  # последний лучший

    return ProdBundle(
        prod_dir=prod_dir,
        meta_path=meta_path,
        feature_cols=feature_cols,
        num_feature_cols=num_cols,
        cat_feature_cols=cat_cols,
        cat_feature_indices=cat_indices,
        x_good=x_good,
        x_bad=x_bad,
        fill_num=fill_num,
        fill_cat=fill_cat,
        score_thresholds_from_oof=score_thresholds_from_oof,
        default_trade_frac=default_trade_frac,
        score_mode=score_mode,
        gate_good=gate_good,
        gate_bad=gate_bad,
        dir_cls=dir_cls,
        pnl_reg=pnl_reg,
    )


# =============================================================================
# PREPARE FEATURES
# =============================================================================

def _prepare_row(bundle: ProdBundle, features: Dict[str, Any], *, strict: bool = True) -> pd.DataFrame:
    """
    Делает DataFrame из 1 строки в нужном порядке колонок и с нужными типами.

    strict=True  -> если каких-то фич нет, падение
    strict=False -> missing заполняем fill_num/fill_cat (или 0/-1)
    """
    row: Dict[str, Any] = {}
    missing: List[str] = []

    for c in bundle.feature_cols:
        if c in features:
            row[c] = features.get(c)
        else:
            missing.append(c)
            row[c] = None

    if missing and strict:
        raise KeyError(f"Missing features: {missing}")

    df = pd.DataFrame([row], columns=bundle.feature_cols)

    # NUM -> float
    for c in bundle.num_feature_cols:
        v_def = bundle.fill_num.get(c, 0.0)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(float(v_def)).astype(float)

    # CAT -> int
    for c in bundle.cat_feature_cols:
        v_def = bundle.fill_cat.get(c, -1)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(int(v_def)).astype(int)

    return df


# =============================================================================
# SCORE
# =============================================================================

def _logit(p: float) -> float:
    eps = 1e-12
    pp = float(p)
    if pp < eps:
        pp = eps
    if pp > 1.0 - eps:
        pp = 1.0 - eps
    return float(np.log(pp / (1.0 - pp)))


def _calc_score(bundle: ProdBundle, p_good: float, p_bad: float) -> float:
    mode = (bundle.score_mode or "").strip().lower()
    if mode == "p_diff":
        return float(p_good - p_bad)
    return float(_logit(p_good) - _logit(p_bad))  # logit_diff


def _pick_threshold(bundle: ProdBundle, trade_frac: float) -> Optional[float]:
    if not bundle.score_thresholds_from_oof:
        return None

    keys = [
        f"{float(trade_frac):.2f}",
        f"{float(trade_frac):.1f}",
        str(trade_frac),
    ]
    for k in keys:
        if k in bundle.score_thresholds_from_oof:
            return float(bundle.score_thresholds_from_oof[k])

    # попытка сравнить численно
    try:
        tf = float(trade_frac)
        for k, v in bundle.score_thresholds_from_oof.items():
            try:
                if abs(float(k) - tf) < 1e-9:
                    return float(v)
            except Exception:
                continue
    except Exception:
        pass

    return None


# =============================================================================
# PREDICT
# =============================================================================

def predict_one(
    bundle: ProdBundle,
    features: Dict[str, Any],
    *,
    trade_frac: Optional[float] = None,
    strict_features: bool = True,
) -> Dict[str, Any]:
    """
    Возвращает:
    - p_good, p_bad, score (score_mode)
    - p_dir1, dir_pred
    - pnl_pred
    - x_good/x_bad
    - trade_decision по порогу из меты (если есть)
    """
    if trade_frac is None:
        trade_frac = bundle.default_trade_frac

    X = _prepare_row(bundle, features, strict=strict_features)
    pool = Pool(X, cat_features=bundle.cat_feature_indices)

    p_good = float(bundle.gate_good.predict_proba(pool)[0][1])
    p_bad = float(bundle.gate_bad.predict_proba(pool)[0][1])
    score = float(_calc_score(bundle, p_good, p_bad))

    p_dir1 = float(bundle.dir_cls.predict_proba(pool)[0][1])
    dir_pred = int(1 if p_dir1 >= 0.5 else 0)

    pnl_pred = float(bundle.pnl_reg.predict(pool)[0])

    score_thr = _pick_threshold(bundle, float(trade_frac))
    trade_decision = None if score_thr is None else bool(score >= float(score_thr))

    return dict(
        p_good=p_good,
        p_bad=p_bad,
        score=score,
        score_mode=bundle.score_mode,
        p_dir1=p_dir1,
        dir_pred=dir_pred,
        pnl_pred=pnl_pred,
        x_good=bundle.x_good,
        x_bad=bundle.x_bad,
        trade_frac=float(trade_frac),
        score_threshold=float(score_thr) if score_thr is not None else None,
        trade_decision=trade_decision,
        prod_dir=bundle.prod_dir,
        meta_path=bundle.meta_path,
    )


def predict_many(
    bundle: ProdBundle,
    rows: List[Dict[str, Any]],
    *,
    trade_frac: Optional[float] = None,
    strict_features: bool = True,
) -> List[Dict[str, Any]]:
    return [predict_one(bundle, r, trade_frac=trade_frac, strict_features=strict_features) for r in rows]


if __name__ == "__main__":
    # Пример (не CLI):
    # prod_dir = "/home/jupiter/PYTHON/T5_ALGO/_models/2025-12-29/185254/catboost_forward_wf_two_stage_best_v2/prod"
    # b = load_prod_bundle(prod_dir)
    # feats = {c: 0 for c in b.feature_cols}
    # print(predict_one(b, feats, strict_features=False))
    pass
