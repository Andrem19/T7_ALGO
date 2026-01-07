# # regime_predict.py
# # -*- coding: utf-8 -*-
# """
# Инференс режимов рынка.

# Требуемый интерфейс (2 функции):
# - load_bundle(...)
# - predict_regime(ohlcv_2d, bundle, timeframe='1h', ...)

# Дополнительно (практично для вашей идеи суперсостояний):
# - predict_superstate(windows_by_tf, bundle, ...)

# Никакого CLI/argparse.
# """

# from __future__ import annotations

# import os
# import json
# from dataclasses import dataclass
# from typing import Dict, Any, Optional, Tuple

# import numpy as np
# import tensorflow as tf


# # =============================================================================
# # CONFIG
# # =============================================================================

# COIN = "BTCUSDT"
# BASE_DIR_DEFAULT = f"_regimes/{COIN}"

# # порядок таймфреймов должен совпадать с prepare_data
# TIMEFRAMES = ["1h", "30m", "15m", "5m"]
# WINDOW_BARS_DEFAULT = 60


# # =============================================================================
# # Feature extraction (должно совпадать с regime_prepare_data.py)
# # =============================================================================

# def featurize_last_window(ohlcv_2d: np.ndarray, window: int) -> np.ndarray:
#     """
#     ohlcv_2d: ndarray shape (N, 6) [ts, o, h, l, c, v]
#     Возвращает X shape (1, window, 6) float32.
#     Берёт последние window баров.
#     """
#     arr = np.asarray(ohlcv_2d, dtype=np.float64)
#     if arr.ndim != 2 or arr.shape[1] < 6:
#         raise ValueError("ohlcv_2d must be 2D array with at least 6 columns [ts,o,h,l,c,v].")
#     if arr.shape[0] < window:
#         raise ValueError(f"Need at least {window} rows, got {arr.shape[0]}.")

#     win = arr[-window:, :6]
#     o = win[:, 1]
#     h = win[:, 2]
#     l = win[:, 3]
#     c = win[:, 4]
#     v = win[:, 5]

#     eps = 1e-9
#     prev_c = np.concatenate([c[:1], c[:-1]], axis=0)
#     base = np.where(np.abs(prev_c) < eps, 1.0, prev_c)

#     f_ret = (c - prev_c) / base
#     f_hl = (h - l) / base
#     f_body = (c - o) / base
#     max_oc = np.maximum(o, c)
#     min_oc = np.minimum(o, c)
#     f_uw = (h - max_oc) / base
#     f_lw = (min_oc - l) / base

#     v_log = np.log1p(np.maximum(v, 0.0))
#     v_mean = v_log.mean()
#     v_std = v_log.std() + eps
#     f_vol = (v_log - v_mean) / v_std

#     X = np.stack([f_ret, f_hl, f_body, f_uw, f_lw, f_vol], axis=-1).astype(np.float32, copy=False)
#     return X[None, :, :]


# def agg_features_from_feats_window(x_win: np.ndarray) -> np.ndarray:
#     """
#     x_win shape (window, 6)
#     Возвращает агрегаты, должны совпадать с prepare_data:
#       [realised_move, realised_vol, wickiness, range_mean]
#     """
#     ret = x_win[:, 0]
#     hl = x_win[:, 1]
#     uw = x_win[:, 3]
#     lw = x_win[:, 4]

#     realised_move = float(ret.sum())
#     realised_vol = float(np.abs(ret).mean())
#     wickiness = float((uw + lw).mean())
#     range_mean = float(hl.mean())

#     return np.array([realised_move, realised_vol, wickiness, range_mean], dtype=np.float32)


# # =============================================================================
# # Bundle
# # =============================================================================

# @dataclass
# class RegimeBundle:
#     base_dir: str
#     config: Dict[str, Any]
#     regime_models: Dict[str, tf.keras.Model]
#     super_model: Optional[tf.keras.Model]


# def load_bundle(base_dir: str = BASE_DIR_DEFAULT) -> RegimeBundle:
#     """
#     Загружает обученные модели и конфиги.
#     """
#     prep_dir = os.path.join(base_dir, "prepared")
#     model_dir = os.path.join(base_dir, "models")

#     cfg_path = os.path.join(prep_dir, "config.json")
#     if not os.path.exists(cfg_path):
#         raise FileNotFoundError(cfg_path)

#     with open(cfg_path, "r", encoding="utf-8") as f:
#         cfg = json.load(f)

#     regime_models: Dict[str, tf.keras.Model] = {}
#     for tf_name in TIMEFRAMES:
#         p = os.path.join(model_dir, f"regime_clf_{tf_name}.keras")
#         if not os.path.exists(p):
#             raise FileNotFoundError(p)
#         regime_models[tf_name] = tf.keras.models.load_model(p, compile=False)

#     super_p = os.path.join(model_dir, "superstate_mlp.keras")
#     super_model = tf.keras.models.load_model(super_p, compile=False) if os.path.exists(super_p) else None

#     return RegimeBundle(
#         base_dir=base_dir,
#         config=cfg,
#         regime_models=regime_models,
#         super_model=super_model,
#     )


# def predict_regime(
#     ohlcv_2d: np.ndarray,
#     bundle: RegimeBundle,
#     timeframe: str = "1h",
#     return_proba: bool = False,
# ) -> int | Tuple[int, np.ndarray]:
#     """
#     Вторая функция интерфейса: принимает 2D ndarray [[ts,o,h,l,c,v], ...] и возвращает режим (class id).
#     """
#     if timeframe not in bundle.regime_models:
#         raise ValueError(f"Unknown timeframe={timeframe}. Available: {list(bundle.regime_models.keys())}")

#     window = int(bundle.config.get("window_bars", WINDOW_BARS_DEFAULT))
#     X = featurize_last_window(ohlcv_2d, window=window)

#     model = bundle.regime_models[timeframe]
#     proba = model.predict(X, verbose=0)[0].astype(np.float32, copy=False)
#     cls = int(np.argmax(proba))

#     if return_proba:
#         return cls, proba
#     return cls


# def _build_meta_vector_from_regimes(
#     regimes: Dict[str, int],
#     n_regimes_by_tf: Dict[str, int],
#     agg: Optional[np.ndarray],
#     meta_cfg: Dict[str, Any],
# ) -> np.ndarray:
#     """
#     Строит meta-вектор в том же порядке, что prepare_data.
#     """
#     blocks = []
#     for tf_name in meta_cfg["timeframes"]:
#         n = int(n_regimes_by_tf[tf_name])
#         r = int(regimes[tf_name])
#         oh = np.zeros((n,), dtype=np.float32)
#         if 0 <= r < n:
#             oh[r] = 1.0
#         blocks.append(oh)

#     if meta_cfg.get("agg_features") and agg is not None:
#         blocks.append(agg.astype(np.float32, copy=False))

#     x_meta = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
#     return x_meta[None, :]


# def predict_superstate(
#     windows_by_tf: Dict[str, np.ndarray],
#     bundle: RegimeBundle,
#     return_proba: bool = False,
# ) -> int | Tuple[int, np.ndarray]:
#     """
#     Практичная функция для вашей схемы: суперсостояние по 4 окнам (4h/1h/15m/5m).
#     Здесь режимы получаем через predict_regime(), затем meta-вектор -> super_model.

#     windows_by_tf: dict { '4h': ndarray2d, '1h': ndarray2d, '15m': ndarray2d, '5m': ndarray2d }
#     """
#     if bundle.super_model is None:
#         raise RuntimeError("superstate model not found (models/superstate_mlp.keras). Train it first.")

#     meta_cfg = bundle.config["meta_cfg"]
#     n_regimes_by_tf = bundle.config["n_regimes_by_tf"]

#     regimes: Dict[str, int] = {}
#     for tf_name in TIMEFRAMES:
#         if tf_name not in windows_by_tf:
#             raise ValueError(f"windows_by_tf must include '{tf_name}'")
#         regimes[tf_name] = int(predict_regime(windows_by_tf[tf_name], bundle, timeframe=tf_name, return_proba=False))

#     # агрегаты считаем из того ТФ, который указан в meta_cfg
#     agg = None
#     agg_tf = meta_cfg.get("agg_tf")
#     if agg_tf:
#         window = int(bundle.config.get("window_bars", WINDOW_BARS_DEFAULT))
#         X_agg = featurize_last_window(windows_by_tf[agg_tf], window=window)[0]  # (window,6)
#         agg = agg_features_from_feats_window(X_agg)

#     x_meta = _build_meta_vector_from_regimes(
#         regimes=regimes,
#         n_regimes_by_tf=n_regimes_by_tf,
#         agg=agg,
#         meta_cfg=meta_cfg,
#     )

#     proba = bundle.super_model.predict(x_meta, verbose=0)[0].astype(np.float32, copy=False)
#     cls = int(np.argmax(proba))
#     if return_proba:
#         return cls, proba
#     return cls, regimes
# regime_predict.py
# -*- coding: utf-8 -*-
"""
Инференс режимов рынка (обновлено под новый devide_classes_1.py).

Поддерживает 2 backend-а:
- "clf"  : keras-классификатор models/regime_clf_<tf>.keras (если вы обучили train_model_1.py)
- "gmm"  : encoder + preprocess + gmm (работает сразу после devide_classes_1.py)
- "auto" : сначала clf, если нет — gmm

Возврат description:
- predict_regime(..., return_description=True) -> (cls, desc) или (cls, proba, desc)
- predict_superstate(..., return_description=True) -> добавит desc суперсостояния

Никакого CLI/argparse.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import tensorflow as tf
import joblib


# =============================================================================
# CONFIG
# =============================================================================

COIN = "BTCUSDT"
BASE_DIR_DEFAULT = f"_regimes/{COIN}"

TIMEFRAMES_DEFAULT = ["1h", "30m", "15m", "5m"]
WINDOW_BARS_DEFAULT = 60


# =============================================================================
# Feature extraction (должно совпадать с devide_classes_1.py)
# =============================================================================

def featurize_last_window(ohlcv_2d: np.ndarray, window: int) -> np.ndarray:
    """
    ohlcv_2d: ndarray shape (N, 6) [ts, o, h, l, c, v]
    Возвращает X shape (1, window, 6) float32.
    Берёт последние window баров.
    """
    arr = np.asarray(ohlcv_2d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError("ohlcv_2d must be 2D array with at least 6 columns [ts,o,h,l,c,v].")
    if arr.shape[0] < window:
        raise ValueError(f"Need at least {window} rows, got {arr.shape[0]}.")

    win = arr[-window:, :6]
    o = win[:, 1]
    h = win[:, 2]
    l = win[:, 3]
    c = win[:, 4]
    v = win[:, 5]

    eps = 1e-9
    prev_c = np.concatenate([c[:1], c[:-1]], axis=0)
    base = np.where(np.abs(prev_c) < eps, 1.0, prev_c)

    f_ret = (c - prev_c) / base
    f_hl = (h - l) / base
    f_body = (c - o) / base
    max_oc = np.maximum(o, c)
    min_oc = np.minimum(o, c)
    f_uw = (h - max_oc) / base
    f_lw = (min_oc - l) / base

    v_log = np.log1p(np.maximum(v, 0.0))
    v_mean = v_log.mean()
    v_std = v_log.std() + eps
    f_vol = (v_log - v_mean) / v_std

    X = np.stack([f_ret, f_hl, f_body, f_uw, f_lw, f_vol], axis=-1).astype(np.float32, copy=False)
    return X[None, :, :]


def agg_features_from_feats_window(x_win: np.ndarray) -> np.ndarray:
    """
    x_win shape (window, 6)
    Возвращает агрегаты, должны совпадать с devide_classes_1.py:
      [realised_move, realised_vol, wickiness, range_mean]
    """
    ret = x_win[:, 0]
    hl = x_win[:, 1]
    uw = x_win[:, 3]
    lw = x_win[:, 4]

    realised_move = float(ret.sum())
    realised_vol = float(np.abs(ret).mean())
    wickiness = float((uw + lw).mean())
    range_mean = float(hl.mean())

    return np.array([realised_move, realised_vol, wickiness, range_mean], dtype=np.float32)


# =============================================================================
# Small helpers
# =============================================================================

def _safe_int_map(d: Any) -> Dict[int, int]:
    """
    mapping из json может быть со строковыми ключами.
    """
    if not isinstance(d, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in d.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out

def _apply_preprocess(x: np.ndarray, pp: Dict[str, Any]) -> np.ndarray:
    """
    pp = {"scaler":..., "pca":...} из joblib.
    """
    X = np.asarray(x, dtype=np.float64)
    scaler = pp.get("scaler")
    pca = pp.get("pca")
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return np.asarray(X, dtype=np.float64)

def _identity_mapping(n: int) -> Dict[int, int]:
    return {i: i for i in range(int(n))}

def _reorder_proba_by_mapping(proba_old: np.ndarray, mapping_old_to_new: Dict[int, int], n_new: int) -> np.ndarray:
    """
    proba от GMM идёт в порядке old label.
    Мы хотим proba в порядке new label.
    """
    p_old = np.asarray(proba_old, dtype=np.float32).reshape(-1)
    p_new = np.zeros((int(n_new),), dtype=np.float32)
    for old_id, p in enumerate(p_old.tolist()):
        new_id = mapping_old_to_new.get(int(old_id))
        if new_id is None:
            continue
        if 0 <= int(new_id) < int(n_new):
            p_new[int(new_id)] = float(p)
    s = float(p_new.sum())
    if s > 0:
        p_new = p_new / s
    return p_new


# =============================================================================
# Bundle
# =============================================================================

@dataclass
class RegimeBundle:
    base_dir: str
    config: Dict[str, Any]

    # descriptions
    desc_regimes: Dict[str, Dict[str, str]]        # tf -> { "0": "...", "1": "..." }
    desc_super: Dict[str, str]                     # { "0": "...", "1": "..." }

    # clf backend
    clf_regime_models: Dict[str, tf.keras.Model]   # tf -> model
    clf_super_model: Optional[tf.keras.Model]

    # gmm backend
    encoders: Dict[str, Optional[tf.keras.Model]]  # tf -> encoder or None
    gmms: Dict[str, Any]                           # tf -> GaussianMixture
    preprocess: Dict[str, Dict[str, Any]]          # tf -> {"scaler":..., "pca":...}
    mapping_old_to_new: Dict[str, Dict[int, int]]  # tf -> mapping

    gmm_super: Optional[Any]
    preprocess_super: Optional[Dict[str, Any]]
    super_mapping_old_to_new: Dict[int, int]


def load_bundle(base_dir: str = BASE_DIR_DEFAULT) -> RegimeBundle:
    """
    Загружает всё, что есть:
    - config + descriptions
    - keras clf models (если обучены)
    - gmm+preprocess+encoder (если есть)
    - gmm_super+preprocess_super (если есть)
    """
    prep_dir = os.path.join(base_dir, "prepared")
    model_dir = os.path.join(base_dir, "models")
    enc_dir = os.path.join(base_dir, "encoders")
    gmm_dir = os.path.join(base_dir, "gmms")
    rep_dir = os.path.join(base_dir, "reports")

    cfg_path = os.path.join(prep_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    timeframes: List[str] = cfg.get("timeframes", TIMEFRAMES_DEFAULT)

    # descriptions (новые)
    desc_block = cfg.get("descriptions", {}) if isinstance(cfg.get("descriptions", {}), dict) else {}
    desc_reg = desc_block.get("regimes", {}) if isinstance(desc_block.get("regimes", {}), dict) else {}
    desc_sup = desc_block.get("superstates", {}) if isinstance(desc_block.get("superstates", {}), dict) else {}

    # clf models (опционально)
    clf_reg: Dict[str, tf.keras.Model] = {}
    for tf_name in timeframes:
        p = os.path.join(model_dir, f"regime_clf_{tf_name}.keras")
        if os.path.exists(p):
            clf_reg[tf_name] = tf.keras.models.load_model(p, compile=False)

    super_p = os.path.join(model_dir, "superstate_mlp.keras")
    clf_super = tf.keras.models.load_model(super_p, compile=False) if os.path.exists(super_p) else None

    # gmm backend
    encoders: Dict[str, Optional[tf.keras.Model]] = {}
    gmms: Dict[str, Any] = {}
    preprocess: Dict[str, Dict[str, Any]] = {}
    mapping_old_to_new: Dict[str, Dict[int, int]] = {}

    for tf_name in timeframes:
        # encoder
        enc_path = os.path.join(enc_dir, f"encoder_{tf_name}.keras")
        encoders[tf_name] = tf.keras.models.load_model(enc_path, compile=False) if os.path.exists(enc_path) else None

        # gmm + preprocess
        gmm_path = os.path.join(gmm_dir, f"gmm_{tf_name}.joblib")
        pp_path = os.path.join(gmm_dir, f"preprocess_{tf_name}.joblib")
        rep_path = os.path.join(rep_dir, f"report_{tf_name}.json")

        if os.path.exists(gmm_path) and os.path.exists(pp_path):
            gmms[tf_name] = joblib.load(gmm_path)
            preprocess[tf_name] = joblib.load(pp_path)

            # mapping old->new хранится в report_<tf>.json
            if os.path.exists(rep_path):
                with open(rep_path, "r", encoding="utf-8") as f:
                    rj = json.load(f)
                mp = _safe_int_map(rj.get("id_mapping_old_to_new", {}))
                mapping_old_to_new[tf_name] = mp
            else:
                # если отчёта нет — считаем идентичность
                n = int(cfg.get("n_regimes_by_tf", {}).get(tf_name, 0) or 0)
                mapping_old_to_new[tf_name] = _identity_mapping(n)
        else:
            # gmm может отсутствовать — это нормально, если вы используете только clf
            pass

    # super gmm backend
    gmm_super = None
    preprocess_super = None
    super_mapping_old_to_new: Dict[int, int] = {}

    gmm_super_path = os.path.join(gmm_dir, "gmm_super.joblib")
    pp_super_path = os.path.join(gmm_dir, "preprocess_super.joblib")
    if os.path.exists(gmm_super_path) and os.path.exists(pp_super_path):
        gmm_super = joblib.load(gmm_super_path)
        preprocess_super = joblib.load(pp_super_path)
        super_mapping_old_to_new = _safe_int_map(preprocess_super.get("mapping_old_to_new", {}))
    else:
        super_mapping_old_to_new = _identity_mapping(int(cfg.get("n_super_states", 0) or 0))

    return RegimeBundle(
        base_dir=base_dir,
        config=cfg,
        desc_regimes=desc_reg,
        desc_super=desc_sup,
        clf_regime_models=clf_reg,
        clf_super_model=clf_super,
        encoders=encoders,
        gmms=gmms,
        preprocess=preprocess,
        mapping_old_to_new=mapping_old_to_new,
        gmm_super=gmm_super,
        preprocess_super=preprocess_super,
        super_mapping_old_to_new=super_mapping_old_to_new,
    )


# =============================================================================
# Description helpers
# =============================================================================

def get_regime_description(bundle: RegimeBundle, timeframe: str, regime_id: int) -> str:
    d_tf = bundle.desc_regimes.get(timeframe, {})
    # ключи в json обычно строки
    s = d_tf.get(str(int(regime_id)))
    return str(s) if s is not None else ""

def get_superstate_description(bundle: RegimeBundle, super_id: int) -> str:
    s = bundle.desc_super.get(str(int(super_id)))
    return str(s) if s is not None else ""


# =============================================================================
# Predict regime
# =============================================================================

def predict_regime(
    ohlcv_2d: np.ndarray,
    bundle: RegimeBundle,
    timeframe: str = "1h",
    return_proba: bool = False,
    return_description: bool = False,
    backend: str = "auto",
) -> int | Tuple:
    """
    Возвращает режим (class id).
    Дополнительно:
      return_proba=True -> добавит proba
      return_description=True -> добавит description

    backend:
      "auto" -> clf если есть, иначе gmm
      "clf"  -> только keras clf
      "gmm"  -> только encoder+preprocess+gmm
    """
    tf_name = timeframe
    cfg = bundle.config
    window = int(cfg.get("window_bars", WINDOW_BARS_DEFAULT))

    # какие режимы считаем "валидным пространством"
    n_regimes_by_tf = cfg.get("n_regimes_by_tf", {}) if isinstance(cfg.get("n_regimes_by_tf", {}), dict) else {}
    n_classes = int(n_regimes_by_tf.get(tf_name, 0) or 0)

    use_clf = False
    if backend not in ("auto", "clf", "gmm"):
        raise ValueError("backend must be 'auto', 'clf' or 'gmm'")

    if backend == "clf":
        use_clf = True
    elif backend == "gmm":
        use_clf = False
    else:
        use_clf = (tf_name in bundle.clf_regime_models)

    X = featurize_last_window(ohlcv_2d, window=window)

    if use_clf:
        if tf_name not in bundle.clf_regime_models:
            raise FileNotFoundError(f"clf model not found for {tf_name}: models/regime_clf_{tf_name}.keras")

        model = bundle.clf_regime_models[tf_name]
        proba = model.predict(X, verbose=0)[0].astype(np.float32, copy=False)
        cls = int(np.argmax(proba))

        desc = get_regime_description(bundle, tf_name, cls) if return_description else None

        if return_proba and return_description:
            return cls, proba, desc
        if return_proba:
            return cls, proba
        if return_description:
            return cls, desc
        return cls

    # --- GMM backend ---
    if tf_name not in bundle.gmms or tf_name not in bundle.preprocess:
        raise FileNotFoundError(f"gmm/preprocess not found for {tf_name}. Run devide_classes_1.py first.")

    gmm = bundle.gmms[tf_name]
    pp = bundle.preprocess[tf_name]
    mp = bundle.mapping_old_to_new.get(tf_name, {})

    encoder = bundle.encoders.get(tf_name)
    if encoder is not None:
        emb_raw = encoder.predict(X, verbose=0)
        emb_raw = np.asarray(emb_raw, dtype=np.float32).reshape(1, -1)
    else:
        # если encoder не сохранён — используем flatten (хуже, но совместимо)
        emb_raw = X.reshape(1, -1).astype(np.float32, copy=False)

    emb = _apply_preprocess(emb_raw, pp)
    cls_old = int(gmm.predict(emb)[0])
    cls = int(mp.get(cls_old, cls_old))

    # proba
    proba_new = None
    if return_proba:
        if hasattr(gmm, "predict_proba"):
            proba_old = gmm.predict_proba(emb)[0]
            if n_classes <= 0:
                n_classes = int(np.asarray(proba_old).size)
            # если mapping пустой — делаем identity
            if not mp and n_classes > 0:
                mp = _identity_mapping(n_classes)
            proba_new = _reorder_proba_by_mapping(proba_old, mp, n_new=int(n_classes))
        else:
            proba_new = np.zeros((max(1, n_classes),), dtype=np.float32)
            if 0 <= cls < proba_new.size:
                proba_new[cls] = 1.0

    desc = get_regime_description(bundle, tf_name, cls) if return_description else None

    if return_proba and return_description:
        return cls, proba_new, desc
    if return_proba:
        return cls, proba_new
    if return_description:
        return cls, desc
    return cls


# =============================================================================
# Superstate helpers
# =============================================================================

def _build_meta_vector_from_regimes(
    regimes: Dict[str, int],
    n_regimes_by_tf: Dict[str, int],
    agg: Optional[np.ndarray],
    meta_cfg: Dict[str, Any],
) -> np.ndarray:
    blocks = []
    for tf_name in meta_cfg["timeframes"]:
        n = int(n_regimes_by_tf[tf_name])
        r = int(regimes[tf_name])
        oh = np.zeros((n,), dtype=np.float32)
        if 0 <= r < n:
            oh[r] = 1.0
        blocks.append(oh)

    if meta_cfg.get("agg_features") and agg is not None:
        blocks.append(agg.astype(np.float32, copy=False))

    x_meta = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    return x_meta[None, :]


# =============================================================================
# Predict superstate
# =============================================================================

def predict_superstate(
    windows_by_tf: Dict[str, np.ndarray],
    bundle: RegimeBundle,
    return_proba: bool = False,
    return_description: bool = False,
    backend: str = "auto",
) -> Tuple:
    """
    Возвращает суперсостояние и словарь режимов по TF.

    Возврат:
      - если return_proba=False и return_description=False:
          (super_cls, regimes_dict)
      - если return_proba=True:
          (super_cls, proba, regimes_dict)
      - если return_description=True:
          добавится desc в конец:
          (super_cls, regimes_dict, desc) или (super_cls, proba, regimes_dict, desc)

    backend:
      "auto" -> super clf если есть, иначе gmm_super
      "clf"  -> только superstate_mlp.keras
      "gmm"  -> только gmm_super.joblib
    """
    cfg = bundle.config
    meta_cfg = cfg.get("meta_cfg", {})
    n_regimes_by_tf = cfg.get("n_regimes_by_tf", {})

    if not isinstance(meta_cfg, dict) or "timeframes" not in meta_cfg:
        raise RuntimeError("meta_cfg not found in config.json")

    timeframes = meta_cfg["timeframes"]

    for tf_name in timeframes:
        if tf_name not in windows_by_tf:
            raise ValueError(f"windows_by_tf must include '{tf_name}'")

    # regimes
    regimes: Dict[str, int] = {}
    for tf_name in timeframes:
        regimes[tf_name] = int(
            predict_regime(
                windows_by_tf[tf_name],
                bundle,
                timeframe=tf_name,
                return_proba=False,
                return_description=False,
                backend="auto",
            )
        )

    # agg (если нужно)
    agg = None
    agg_tf = meta_cfg.get("agg_tf")
    if agg_tf:
        window = int(cfg.get("window_bars", WINDOW_BARS_DEFAULT))
        X_agg = featurize_last_window(windows_by_tf[agg_tf], window=window)[0]
        agg = agg_features_from_feats_window(X_agg)

    x_meta = _build_meta_vector_from_regimes(
        regimes=regimes,
        n_regimes_by_tf=n_regimes_by_tf,
        agg=agg,
        meta_cfg=meta_cfg,
    )

    # choose backend
    if backend not in ("auto", "clf", "gmm"):
        raise ValueError("backend must be 'auto', 'clf' or 'gmm'")

    use_clf = False
    if backend == "clf":
        use_clf = True
    elif backend == "gmm":
        use_clf = False
    else:
        use_clf = (bundle.clf_super_model is not None)

    proba = None
    cls = None

    if use_clf:
        if bundle.clf_super_model is None:
            raise FileNotFoundError("super clf model not found: models/superstate_mlp.keras")
        proba = bundle.clf_super_model.predict(x_meta, verbose=0)[0].astype(np.float32, copy=False)
        cls = int(np.argmax(proba))
    else:
        if bundle.gmm_super is None or bundle.preprocess_super is None:
            raise FileNotFoundError("gmm_super/preprocess_super not found. Run devide_classes_1.py (and it must finish super).")

        gmm = bundle.gmm_super
        pp = bundle.preprocess_super
        mp = bundle.super_mapping_old_to_new or {}

        x_proc = _apply_preprocess(x_meta, pp)
        cls_old = int(gmm.predict(x_proc)[0])
        cls = int(mp.get(cls_old, cls_old))

        if return_proba:
            if hasattr(gmm, "predict_proba"):
                p_old = gmm.predict_proba(x_proc)[0]
                n_new = int(cfg.get("n_super_states", len(p_old)) or len(p_old))
                if not mp and n_new > 0:
                    mp = _identity_mapping(n_new)
                proba = _reorder_proba_by_mapping(p_old, mp, n_new=n_new)
            else:
                n_new = int(cfg.get("n_super_states", 1) or 1)
                proba = np.zeros((max(1, n_new),), dtype=np.float32)
                if 0 <= cls < proba.size:
                    proba[cls] = 1.0

    desc = get_superstate_description(bundle, int(cls)) if return_description else None

    if return_proba and return_description:
        return int(cls), proba, regimes, desc
    if return_proba:
        return int(cls), proba, regimes
    if return_description:
        return int(cls), regimes, desc
    return int(cls), regimes
