#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hmm_regime_model.py

Автономная Hidden Markov Model (Gaussian, diag covariance) для выделения режимов BTC по 1D свечам.

Формат CSV по умолчанию (как у вас): БЕЗ хедера, колонки по порядку:
  timestamp_ms, open, high, low, close, volume

Возможности:
- train_hmm_regime_model(...): обучить/перетренировать модель на заданном периоде и сохранить на диск
  + сохранить артефакты: states/probs CSV, transition matrix, summary, loglik, картинки
- load_hmm_regime_model(...): загрузить сохранённую модель в память
- infer_regime_latest(...): прод-функция: дать последние N свечей -> получить текущее состояние и вероятности

Примечания:
- Никакого argparse, параметры только через функции/константы.
- Модель и артефакты сохраняются в out_dir (по умолчанию рядом с model_path).
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Default config (можно переопределять параметрами функций)
# =============================================================================

DEFAULT_N_STATES = 4
DEFAULT_N_ITER = 250
DEFAULT_TOL = 1e-4
DEFAULT_SEED = 42

DEFAULT_WIN_VOL = 20
DEFAULT_WIN_SMA_FAST = 20
DEFAULT_WIN_SMA_SLOW = 80

DEFAULT_INIT_SELF_TRANS = 0.95

MIN_COVAR = 1e-6
MIN_PROB = 1e-12


# =============================================================================
# Helpers
# =============================================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_utc_ts(x: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    ts = pd.to_datetime(x, utc=True)
    return ts


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, MIN_PROB, None))


def _logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _expected_duration_days(p_stay: float) -> float:
    # “сколько дней в среднем длится режим” = один разделить на (один минус вероятность остаться)
    return 1.0 / max(1.0 - float(p_stay), 1e-12)


# =============================================================================
# IO: read CSV (no header by default)
# =============================================================================

def read_ohlcv_csv(
    csv_path: str,
    *,
    has_header: bool = False,
) -> pd.DataFrame:
    """
    Читает CSV со свечами.
    По умолчанию ожидаем CSV без хедера: [ts_ms, open, high, low, close, volume].
    Возвращает df с колонками: ts_ms, open, high, low, close, volume, dt (UTC).
    """
    if has_header:
        df = pd.read_csv(csv_path)
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

        # пробуем найти колонки по именам
        # допускаем варианты: timestamp_ms / ts_ms / time_ms; open/high/low/close/volume
        def pick(cands: List[str]) -> Optional[str]:
            for c in cands:
                if c in df.columns:
                    return c
            return None

        c_ts = pick(["timestamp_ms", "ts_ms", "time_ms", "timestamp", "time"])
        c_o = pick(["open", "o"])
        c_h = pick(["high", "h"])
        c_l = pick(["low", "l"])
        c_c = pick(["close", "c"])
        c_v = pick(["volume", "vol", "v"])

        miss = [("ts", c_ts), ("open", c_o), ("high", c_h), ("low", c_l), ("close", c_c)]
        miss2 = [k for k, v in miss if v is None]
        if miss2:
            raise ValueError(f"Не найдены обязательные колонки: {miss2}. Найдены: {list(df.columns)}")

        out = pd.DataFrame({
            "ts_ms": pd.to_numeric(df[c_ts], errors="coerce"),
            "open": pd.to_numeric(df[c_o], errors="coerce"),
            "high": pd.to_numeric(df[c_h], errors="coerce"),
            "low": pd.to_numeric(df[c_l], errors="coerce"),
            "close": pd.to_numeric(df[c_c], errors="coerce"),
            "volume": pd.to_numeric(df[c_v], errors="coerce") if c_v is not None else 0.0,
        })
    else:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 6:
            raise ValueError(f"Ожидаю минимум 6 колонок, получили {df.shape[1]}")
        df = df.iloc[:, :6].copy()
        df.columns = ["ts_ms", "open", "high", "low", "close", "volume"]

        out = pd.DataFrame()
        for c in ["ts_ms", "open", "high", "low", "close", "volume"]:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    out = out.dropna(subset=["ts_ms", "open", "high", "low", "close"]).copy()
    out["ts_ms"] = out["ts_ms"].astype("int64")
    out["dt"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True)
    out = out.sort_values("dt").reset_index(drop=True)
    return out


def slice_period(
    df: pd.DataFrame,
    *,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Оставляет строки df, где start <= dt < end.
    """
    s = _to_utc_ts(start)
    e = _to_utc_ts(end)
    m = np.ones(len(df), dtype=bool)
    if s is not None:
        m &= (df["dt"] >= s)
    if e is not None:
        m &= (df["dt"] < e)
    return df.loc[m].copy().reset_index(drop=True)


# =============================================================================
# Features
# =============================================================================

@dataclass
class FeatureConfig:
    win_vol: int = DEFAULT_WIN_VOL
    win_sma_fast: int = DEFAULT_WIN_SMA_FAST
    win_sma_slow: int = DEFAULT_WIN_SMA_SLOW

    def as_dict(self) -> Dict[str, Any]:
        return {
            "win_vol": int(self.win_vol),
            "win_sma_fast": int(self.win_sma_fast),
            "win_sma_slow": int(self.win_sma_slow),
        }


def build_features_1d(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    mu: Optional[np.ndarray] = None,
    sd: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Строит признаки и возвращает:
      df_feat: выровненный df (без NaN, отсортирован)
      X_std: стандартизированные признаки
      feat_cols: имена признаков
      mu, sd: статистики стандартизации (если mu/sd передали, возвращаем их же)
    """
    d = df.copy()

    # дневное изменение close
    d["ret_1d"] = d["close"].pct_change()

    # дневной диапазон относительно цены
    d["range_1d"] = (d["high"] - d["low"]) / d["close"].replace(0.0, np.nan)

    # локальная волатильность по доходностям
    d["vol_roll"] = d["ret_1d"].rolling(int(cfg.win_vol)).std()

    sma_fast = d["close"].rolling(int(cfg.win_sma_fast)).mean()
    sma_slow = d["close"].rolling(int(cfg.win_sma_slow)).mean()

    # положение цены относительно медленной средней
    d["trend_slow"] = (d["close"] - sma_slow) / sma_slow.replace(0.0, np.nan)

    # спред быстрый/медленный
    d["ma_spread"] = (sma_fast - sma_slow) / sma_slow.replace(0.0, np.nan)

    d = d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feat_cols = ["ret_1d", "range_1d", "vol_roll", "trend_slow", "ma_spread"]
    F = d[feat_cols].astype("float64").values

    if mu is None or sd is None:
        mu2 = np.nanmean(F, axis=0)
        sd2 = np.nanstd(F, axis=0)
        sd2 = np.where(sd2 < 1e-12, 1.0, sd2)
        mu, sd = mu2, sd2
    else:
        mu = np.asarray(mu, dtype=np.float64)
        sd = np.asarray(sd, dtype=np.float64)
        if mu.shape[0] != F.shape[1] or sd.shape[0] != F.shape[1]:
            raise ValueError("mu/sd не совпадают по размерности с числом признаков.")

    X_std = (F - mu) / sd
    return d, X_std, feat_cols, mu, sd


# =============================================================================
# Gaussian HMM (diag covariance) - EM / Baum-Welch
# =============================================================================

@dataclass
class HMMParams:
    startprob: np.ndarray   # (K,)
    transmat: np.ndarray    # (K,K)
    means: np.ndarray       # (K,D)
    vars: np.ndarray        # (K,D)


class GaussianHMMDiag:
    """
    Диагональная Gaussian HMM:
    - fit(X): EM
    - predict_proba_smooth(X): posterior p(z_t | x_1..x_T) (forward-backward)
    - filter_proba(X): filtered p(z_t | x_1..x_t) (forward only)
    - filtered_last(X): p(z_T | x_1..x_T) (каузально на последнем дне)
    - viterbi(X): наиболее вероятная цепочка состояний
    """

    def __init__(self, n_states: int, *, n_iter: int, tol: float, seed: int, init_self_trans: float) -> None:
        self.K = int(n_states)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.rng = np.random.default_rng(int(seed))
        self.init_self_trans = float(init_self_trans)

        self.params: Optional[HMMParams] = None
        self.loglik_history: List[float] = []

    def _init_params(self, X: np.ndarray) -> None:
        T, D = X.shape
        x0 = X[:, 0]
        qs = np.quantile(x0, np.linspace(0.0, 1.0, self.K + 1))

        means = np.zeros((self.K, D), dtype=np.float64)
        vars_ = np.zeros((self.K, D), dtype=np.float64)

        for k in range(self.K):
            mask = (x0 >= qs[k]) & (x0 <= qs[k + 1])
            if mask.sum() < 10:
                idx = self.rng.integers(0, T, size=min(80, T))
                chunk = X[idx]
            else:
                chunk = X[mask]
            means[k] = np.mean(chunk, axis=0)
            vars_[k] = np.maximum(np.var(chunk, axis=0) + MIN_COVAR, MIN_COVAR)

        startprob = np.ones(self.K, dtype=np.float64) / self.K

        transmat = np.ones((self.K, self.K), dtype=np.float64) * ((1.0 - self.init_self_trans) / max(self.K - 1, 1))
        np.fill_diagonal(transmat, self.init_self_trans)

        self.params = HMMParams(startprob=startprob, transmat=transmat, means=means, vars=vars_)

    def _log_emission_prob(self, X: np.ndarray) -> np.ndarray:
        assert self.params is not None
        means = self.params.means
        vars_ = np.maximum(self.params.vars, MIN_COVAR)

        logB = np.zeros((X.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            diff = X - means[k]
            var = vars_[k]
            log_det = np.sum(np.log(2.0 * math.pi * var))
            quad = np.sum((diff * diff) / var, axis=1)
            logB[:, k] = -0.5 * (log_det + quad)
        return logB

    def _forward_log_scaled(
        self,
        logB: np.ndarray,
        *,
        startprob_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward в log-space со скейлингом:
          log_alpha[t] нормирован так, что сумма exp(log_alpha[t]) == 1
          c[t] = log-нормировщик
          loglik = сумма c
        """
        assert self.params is not None
        startprob = self.params.startprob if startprob_override is None else np.asarray(startprob_override, dtype=np.float64)
        startprob = startprob / max(float(np.sum(startprob)), MIN_PROB)

        log_start = _safe_log(startprob)
        log_trans = _safe_log(self.params.transmat)

        T, K = logB.shape
        log_alpha = np.zeros((T, K), dtype=np.float64)
        c = np.zeros(T, dtype=np.float64)

        log_alpha[0] = log_start + logB[0]
        c[0] = _logsumexp(log_alpha[0], axis=0)
        log_alpha[0] -= c[0]

        for t in range(1, T):
            prev = log_alpha[t - 1].reshape(K, 1) + log_trans
            log_alpha[t] = logB[t] + _logsumexp(prev, axis=0)
            c[t] = _logsumexp(log_alpha[t], axis=0)
            log_alpha[t] -= c[t]

        return log_alpha, c, float(np.sum(c))

    def _backward_log_scaled(self, logB: np.ndarray, c: np.ndarray) -> np.ndarray:
        assert self.params is not None
        log_trans = _safe_log(self.params.transmat)

        T, K = logB.shape
        log_beta = np.zeros((T, K), dtype=np.float64)
        log_beta[T - 1] = 0.0

        for t in range(T - 2, -1, -1):
            nxt = log_trans + (logB[t + 1] + log_beta[t + 1]).reshape(1, K)
            log_beta[t] = _logsumexp(nxt, axis=1)
            # согласуем со скейлингом forward: вычитаем нормировщик следующего шага
            log_beta[t] -= c[t + 1]

        return log_beta

    def _posteriors(
        self,
        X: np.ndarray,
        *,
        startprob_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Возвращает:
          gamma: posterior p(z_t=k | x_1..x_T)
          xi_sum: сумма ожидаемых переходов
          loglik
        """
        assert self.params is not None
        logB = self._log_emission_prob(X)
        log_alpha, c, loglik = self._forward_log_scaled(logB, startprob_override=startprob_override)
        log_beta = self._backward_log_scaled(logB, c)

        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # xi_sum
        log_trans = _safe_log(self.params.transmat)
        T, K = logB.shape
        xi_sum = np.zeros((K, K), dtype=np.float64)
        for t in range(T - 1):
            log_xi = (log_alpha[t].reshape(K, 1) +
                      log_trans +
                      (logB[t + 1] + log_beta[t + 1]).reshape(1, K))
            log_xi -= _logsumexp(log_xi, axis=None)
            xi_sum += np.exp(log_xi)

        return gamma, xi_sum, loglik

    def fit(self, X: np.ndarray) -> "GaussianHMMDiag":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 200:
            raise ValueError("X должен быть 2D и иметь достаточно строк для обучения.")

        self._init_params(X)

        prev_ll = None
        for _it in range(self.n_iter):
            gamma, xi_sum, ll = self._posteriors(X)
            self.loglik_history.append(ll)

            assert self.params is not None
            K, D = self.K, X.shape[1]

            # startprob
            startprob = gamma[0].copy()
            startprob = startprob / max(float(np.sum(startprob)), MIN_PROB)

            # transmat
            trans = xi_sum.copy()
            row_sum = np.sum(trans, axis=1, keepdims=True)
            row_sum = np.where(row_sum < MIN_PROB, MIN_PROB, row_sum)
            trans = trans / row_sum

            # means & vars
            w_sum = np.sum(gamma, axis=0)
            w_sum = np.where(w_sum < MIN_PROB, MIN_PROB, w_sum)

            means = (gamma.T @ X) / w_sum.reshape(K, 1)

            vars_ = np.zeros((K, D), dtype=np.float64)
            for k in range(K):
                diff = X - means[k]
                vars_[k] = (gamma[:, k].reshape(-1, 1) * (diff * diff)).sum(axis=0) / w_sum[k]
            vars_ = np.maximum(vars_ + MIN_COVAR, MIN_COVAR)

            self.params = HMMParams(startprob=startprob, transmat=trans, means=means, vars=vars_)

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def predict_proba_smooth(self, X: np.ndarray, *, startprob_override: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Posterior по каждому дню, использует forward-backward по всему окну.
        На последнем дне это совпадает с каузальной оценкой.
        """
        gamma, _xi_sum, _ll = self._posteriors(np.asarray(X, dtype=np.float64), startprob_override=startprob_override)
        return gamma

    def filter_proba(self, X: np.ndarray, *, startprob_override: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Каузальная оценка по дням: p(z_t | x_1..x_t) (forward only).
        Возвращает вероятности по каждому t.
        """
        X = np.asarray(X, dtype=np.float64)
        logB = self._log_emission_prob(X)
        log_alpha, _c, _ll = self._forward_log_scaled(logB, startprob_override=startprob_override)
        probs = np.exp(log_alpha)
        probs = probs / np.maximum(np.sum(probs, axis=1, keepdims=True), MIN_PROB)
        return probs

    def filtered_last(self, X: np.ndarray, *, startprob_override: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Каузальная оценка на последнем дне: p(z_T | x_1..x_T)
        """
        probs = self.filter_proba(X, startprob_override=startprob_override)
        return probs[-1].copy()

    def viterbi(self, X: np.ndarray, *, startprob_override: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.params is not None
        X = np.asarray(X, dtype=np.float64)
        logB = self._log_emission_prob(X)

        startprob = self.params.startprob if startprob_override is None else np.asarray(startprob_override, dtype=np.float64)
        startprob = startprob / max(float(np.sum(startprob)), MIN_PROB)

        log_start = _safe_log(startprob)
        log_trans = _safe_log(self.params.transmat)

        T, K = logB.shape
        delta = np.zeros((T, K), dtype=np.float64)
        psi = np.zeros((T, K), dtype=np.int32)

        delta[0] = log_start + logB[0]
        for t in range(1, T):
            scores = delta[t - 1].reshape(K, 1) + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + np.max(scores, axis=0)

        states = np.zeros(T, dtype=np.int32)
        states[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])
        return states


# =============================================================================
# Model container + save/load
# =============================================================================

@dataclass
class HMMRegimeModel:
    """
    Обёртка для прод-использования.
    """
    n_states: int
    feat_cols: List[str]
    feature_cfg: FeatureConfig
    mu: np.ndarray
    sd: np.ndarray
    params: HMMParams

    def make_hmm(self, *, n_iter: int = 1, tol: float = DEFAULT_TOL, seed: int = DEFAULT_SEED) -> GaussianHMMDiag:
        hmm = GaussianHMMDiag(self.n_states, n_iter=n_iter, tol=tol, seed=seed, init_self_trans=DEFAULT_INIT_SELF_TRANS)
        hmm.params = self.params
        return hmm


def save_hmm_regime_model(model: HMMRegimeModel, model_path: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Сохраняет модель в .npz (без pickle).
    """
    meta2 = dict(meta or {})
    meta2.update({
        "n_states": int(model.n_states),
        "feat_cols": list(model.feat_cols),
        "feature_cfg": model.feature_cfg.as_dict(),
    })
    meta_json = json.dumps(meta2, ensure_ascii=False)

    np.savez_compressed(
        model_path,
        startprob=model.params.startprob.astype(np.float64),
        transmat=model.params.transmat.astype(np.float64),
        means=model.params.means.astype(np.float64),
        vars=model.params.vars.astype(np.float64),
        mu=model.mu.astype(np.float64),
        sd=model.sd.astype(np.float64),
        meta_json=np.array(meta_json, dtype=object),
    )


def load_hmm_regime_model(model_path: str) -> HMMRegimeModel:
    """
    Загружает модель из .npz
    """
    z = np.load(model_path, allow_pickle=True)
    meta_json = str(z["meta_json"].item())
    meta = json.loads(meta_json)

    n_states = int(meta["n_states"])
    feat_cols = list(meta["feat_cols"])
    fc = meta.get("feature_cfg", {})
    feature_cfg = FeatureConfig(
        win_vol=int(fc.get("win_vol", DEFAULT_WIN_VOL)),
        win_sma_fast=int(fc.get("win_sma_fast", DEFAULT_WIN_SMA_FAST)),
        win_sma_slow=int(fc.get("win_sma_slow", DEFAULT_WIN_SMA_SLOW)),
    )

    params = HMMParams(
        startprob=np.asarray(z["startprob"], dtype=np.float64),
        transmat=np.asarray(z["transmat"], dtype=np.float64),
        means=np.asarray(z["means"], dtype=np.float64),
        vars=np.asarray(z["vars"], dtype=np.float64),
    )
    mu = np.asarray(z["mu"], dtype=np.float64)
    sd = np.asarray(z["sd"], dtype=np.float64)

    return HMMRegimeModel(
        n_states=n_states,
        feat_cols=feat_cols,
        feature_cfg=feature_cfg,
        mu=mu,
        sd=sd,
        params=params,
    )


# =============================================================================
# Artifacts: summaries, segments, plots
# =============================================================================

def _make_segments(df_states: pd.DataFrame) -> pd.DataFrame:
    """
    Превращает дневной state в сегменты [start..end] по непрерывным кускам.
    df_states должен содержать: dt, state
    """
    dt = df_states["dt"].values
    st = df_states["state"].values.astype(int)

    rows = []
    if len(st) == 0:
        return pd.DataFrame(columns=["start_dt", "end_dt", "state", "days"])

    seg_start = 0
    for i in range(1, len(st)):
        if st[i] != st[i - 1]:
            rows.append((dt[seg_start], dt[i - 1], int(st[seg_start]), int(i - seg_start)))
            seg_start = i
    rows.append((dt[seg_start], dt[len(st) - 1], int(st[seg_start]), int(len(st) - seg_start)))

    return pd.DataFrame(rows, columns=["start_dt", "end_dt", "state", "days"])


def _state_summary(df_states: pd.DataFrame, n_states: int, transmat: np.ndarray) -> pd.DataFrame:
    tmp = df_states.copy()
    tmp["ret_1d"] = tmp["close"].pct_change()
    tmp["range_1d"] = (tmp["high"] - tmp["low"]) / tmp["close"].replace(0.0, np.nan)

    rows = []
    for k in range(n_states):
        m = tmp["state"] == k
        n = int(m.sum())
        if n < 5:
            rows.append((k, n, np.nan, np.nan, np.nan, float(transmat[k, k]), _expected_duration_days(float(transmat[k, k]))))
            continue
        avg_ret = float(np.nanmean(tmp.loc[m, "ret_1d"]))
        std_ret = float(np.nanstd(tmp.loc[m, "ret_1d"]))
        avg_rng = float(np.nanmean(tmp.loc[m, "range_1d"]))
        p_stay = float(transmat[k, k])
        rows.append((k, n, avg_ret, std_ret, avg_rng, p_stay, _expected_duration_days(p_stay)))

    out = pd.DataFrame(
        rows,
        columns=["state", "days", "avg_ret_1d", "std_ret_1d", "avg_range_1d", "p_stay", "est_duration_days"]
    ).sort_values("avg_ret_1d", ascending=False, na_position="last").reset_index(drop=True)
    return out


def plot_regime_timeline(df_states: pd.DataFrame, out_png: str, title: str) -> None:
    dt = pd.to_datetime(df_states["dt"], utc=True).dt.tz_convert(None)
    close = df_states["close"].astype(float).values
    states = df_states["state"].astype(int).values

    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.plot(dt, close)
    ax1.set_title(title)
    ax1.set_ylabel("Close")

    ax2.step(dt, states, where="post")
    ax2.set_ylabel("State id")
    ax2.set_xlabel("Date")

    changes = np.where(states[1:] != states[:-1])[0] + 1
    for idx in changes:
        ax1.axvline(dt.iloc[idx], linewidth=0.7, alpha=0.55)
        ax2.axvline(dt.iloc[idx], linewidth=0.7, alpha=0.55)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_regime_probabilities(df_states: pd.DataFrame, n_states: int, out_png: str, title: str) -> None:
    dt = pd.to_datetime(df_states["dt"], utc=True).dt.tz_convert(None)

    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(1, 1, 1)

    for k in range(n_states):
        col = f"prob_state_{k}"
        if col in df_states.columns:
            ax.plot(dt, df_states[col].astype(float).values, label=f"state_{k}")

    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# =============================================================================
# TRAIN / RETRAIN
# =============================================================================

def train_hmm_regime_model(
    csv_path: str,
    model_path: str,
    *,
    out_dir: Optional[str] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    n_states: int = DEFAULT_N_STATES,
    n_iter: int = DEFAULT_N_ITER,
    tol: float = DEFAULT_TOL,
    seed: int = DEFAULT_SEED,
    has_header: bool = False,
    feature_cfg: Optional[FeatureConfig] = None,
    init_self_trans: float = DEFAULT_INIT_SELF_TRANS,
    save_zoom_last_year: bool = True,
) -> Dict[str, str]:
    """
    Обучает HMM на свечах из csv_path и сохраняет:
      - модель: model_path
      - артефакты: out_dir/*.csv + *.png

    Период обучения:
      start <= dt < end

    ВАЖНО (изменение по задаче):
    - timeline и train_states (regime_states.csv / train_states.csv) строятся ДО КОНЦА ВСЕХ ДАННЫХ:
      * после end модель НЕ обучаем
      * после end только инференс обученной моделью
    """
    if feature_cfg is None:
        feature_cfg = FeatureConfig()

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(model_path)) or "."

    _ensure_dir(out_dir)

    # --- читаем полный CSV один раз ---
    df_all = read_ohlcv_csv(csv_path, has_header=has_header)

    # --- train slice ---
    df_train_raw = slice_period(df_all, start=start, end=end)

    if len(df_train_raw) < 500:
        raise ValueError("Слишком мало данных для обучения. Нужны сотни/тысячи дневных свечей.")

    df_train_feat, X_train, feat_cols, mu, sd = build_features_1d(df_train_raw, feature_cfg)

    if len(df_train_feat) < 300:
        raise ValueError("После расчёта признаков осталось слишком мало строк. Увеличьте период данных.")

    # --- train HMM ---
    hmm = GaussianHMMDiag(
        n_states=int(n_states),
        n_iter=int(n_iter),
        tol=float(tol),
        seed=int(seed),
        init_self_trans=float(init_self_trans),
    ).fit(X_train)

    assert hmm.params is not None

    # =========================================================================
    # (1) TRAIN-артефакты (как раньше): states/probs по TRAIN-периоду
    # =========================================================================
    probs_train_smooth = hmm.predict_proba_smooth(X_train)
    states_train = hmm.viterbi(X_train)

    out_states_train = df_train_feat[["dt", "ts_ms", "open", "high", "low", "close", "volume"]].copy()
    out_states_train["state"] = states_train.astype(int)
    for k in range(int(n_states)):
        out_states_train[f"prob_state_{k}"] = probs_train_smooth[:, k]

    # =========================================================================
    # save model (как раньше)
    # =========================================================================
    model = HMMRegimeModel(
        n_states=int(n_states),
        feat_cols=feat_cols,
        feature_cfg=feature_cfg,
        mu=mu,
        sd=sd,
        params=hmm.params,
    )
    meta = {
        "trained_on_csv": os.path.abspath(csv_path),
        "train_start": str(_to_utc_ts(start) if start is not None else df_train_raw["dt"].min()),
        "train_end": str(_to_utc_ts(end) if end is not None else df_train_raw["dt"].max()),
        "rows_raw": int(len(df_train_raw)),
        "rows_feat": int(len(df_train_feat)),
        "n_iter_used": int(len(hmm.loglik_history)),
        "seed": int(seed),
        "init_self_trans": float(init_self_trans),
    }
    save_hmm_regime_model(model, model_path, meta=meta)

    # =========================================================================
    # (2) FULL timeline + FULL train_states.csv (по задаче):
    #     после end => только инференс обученной моделью
    # =========================================================================
    out_states_full = out_states_train

    end_ts = _to_utc_ts(end) if end is not None else None
    if end_ts is not None:
        # берём данные от start до конца файла (end=None)
        df_full_raw = slice_period(df_all, start=start, end=None)

        # ВАЖНО: фичи строим на полном периоде (чтобы роллинги продолжались после end),
        # но стандартизацию делаем mu/sd, посчитанными на TRAIN.
        df_full_feat, X_full_std, _feat_cols_full, _mu2, _sd2 = build_features_1d(
            df_full_raw, feature_cfg, mu=mu, sd=sd
        )

        m_oos = (df_full_feat["dt"] >= end_ts).values
        if bool(np.any(m_oos)):
            X_oos = X_full_std[m_oos]

            # стартовое распределение для oos-цепочки берём каузально из TRAIN
            startprob_oos = hmm.filtered_last(X_train)

            # чтобы не “переехать” train-часть (и не менять её),
            # OOS считаем отдельно по сегменту [end..конец], начиная с startprob_oos
            probs_oos = hmm.predict_proba_smooth(X_oos, startprob_override=startprob_oos)
            states_oos = hmm.viterbi(X_oos, startprob_override=startprob_oos)

            out_states_oos = df_full_feat.loc[m_oos, ["dt", "ts_ms", "open", "high", "low", "close", "volume"]].copy()
            out_states_oos["state"] = states_oos.astype(int)
            for k in range(int(n_states)):
                out_states_oos[f"prob_state_{k}"] = probs_oos[:, k]

            out_states_full = pd.concat([out_states_train, out_states_oos], ignore_index=True)

    # =========================================================================
    # save artefacts
    # =========================================================================

    # --- FULL states CSV (важно: теперь полный период, включая после end) ---
    states_csv = os.path.join(out_dir, "regime_states.csv")
    out_states_full.to_csv(states_csv, index=False)

    # Дополнительно — под ваше имя файла (если вы на него завязаны)
    train_states_csv = os.path.join(out_dir, "train_states.csv")
    if os.path.abspath(train_states_csv) != os.path.abspath(states_csv):
        out_states_full.to_csv(train_states_csv, index=False)

    # transition matrix (как раньше)
    trans_df = pd.DataFrame(hmm.params.transmat, columns=[f"to_{i}" for i in range(int(n_states))])
    trans_df.insert(0, "from_state", [f"from_{i}" for i in range(int(n_states))])
    trans_csv = os.path.join(out_dir, "transition_matrix.csv")
    trans_df.to_csv(trans_csv, index=False)

    # summary / segments (как раньше, по TRAIN)
    summary_df = _state_summary(out_states_train, int(n_states), hmm.params.transmat)
    summary_csv = os.path.join(out_dir, "state_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    seg_df = _make_segments(out_states_train[["dt", "state"]].copy())
    seg_csv = os.path.join(out_dir, "segments.csv")
    seg_df.to_csv(seg_csv, index=False)

    # loglik (как раньше)
    loglik_csv = os.path.join(out_dir, "loglik_history.csv")
    pd.DataFrame({"iter": np.arange(len(hmm.loglik_history)), "loglik": hmm.loglik_history}).to_csv(loglik_csv, index=False)

    # --- timeline (важно: теперь FULL, включая после end) ---
    timeline_png = os.path.join(out_dir, "regime_timeline.png")
    plot_regime_timeline(out_states_full, timeline_png, title="HMM regimes timeline (train + inferred after end)")

    # Дополнительно — под ваше имя файла
    timeline_price_and_regimes_png = os.path.join(out_dir, "timeline_price_and_regimes.png")
    if os.path.abspath(timeline_price_and_regimes_png) != os.path.abspath(timeline_png):
        plot_regime_timeline(out_states_full, timeline_price_and_regimes_png, title="HMM regimes timeline (train + inferred after end)")

    # probabilities plot (как раньше, по TRAIN)
    probs_png = os.path.join(out_dir, "regime_probabilities.png")
    plot_regime_probabilities(out_states_train, int(n_states), probs_png, title="HMM posterior probabilities (train period)")

    # zoom last year (как раньше, по TRAIN)
    zoom_timeline_png = ""
    zoom_probs_png = ""
    if save_zoom_last_year:
        if len(out_states_train) > 400:
            tail = out_states_train.tail(365).copy()
        else:
            tail = out_states_train.copy()

        zoom_timeline_png = os.path.join(out_dir, "regime_timeline_last_year.png")
        plot_regime_timeline(tail, zoom_timeline_png, title="HMM regimes timeline (last ~365 days of train)")

        zoom_probs_png = os.path.join(out_dir, "regime_probabilities_last_year.png")
        plot_regime_probabilities(tail, int(n_states), zoom_probs_png, title="HMM posterior probabilities (last ~365 days of train)")

    return {
        "model_path": model_path,
        "out_dir": out_dir,
        "regime_states_csv": states_csv,
        "transition_matrix_csv": trans_csv,
        "state_summary_csv": summary_csv,
        "segments_csv": seg_csv,
        "loglik_history_csv": loglik_csv,
        "regime_timeline_png": timeline_png,
        "regime_probabilities_png": probs_png,
        "regime_timeline_last_year_png": zoom_timeline_png,
        "regime_probabilities_last_year_png": zoom_probs_png,
    }



# =============================================================================
# PROD inference
# =============================================================================

CandleArray = Union[np.ndarray, List[List[float]], List[Tuple[float, float, float, float, float, float]]]


def infer_regime_latest(
    candles_1d: CandleArray,
    model: HMMRegimeModel,
    *,
    lookback_days: int = 180,
    return_history: bool = True,
    causal_probs: bool = True,
) -> Dict[str, Any]:
    """
    Production-функция.

    Вход:
      candles_1d: последние дневные свечи в формате:
        [[ts_ms, open, high, low, close, volume], ...]
      model: загруженная HMMRegimeModel
      lookback_days: сколько последних свечей использовать (если данных больше)
      return_history: вернуть ли историю states/probs по окну
      causal_probs: если True, вероятности по дням будут каузальные (только прошлое)
                   если False, будут сглаженные (forward-backward по окну)

    Выход:
      {
        "asof_dt": datetime последней свечи,
        "current_state": int,
        "current_state_probs": np.ndarray shape (K,),
        "dt": np.ndarray datetime по окну (если return_history),
        "close": np.ndarray (если return_history),
        "states": np.ndarray (Viterbi) (если return_history),
        "probs": np.ndarray (T,K) (если return_history),
      }
    """
    arr = np.asarray(candles_1d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError("candles_1d должен быть 2D массивом с 6 колонками: ts_ms, open, high, low, close, volume")

    # берём последние lookback_days
    if lookback_days is not None and lookback_days > 0 and arr.shape[0] > lookback_days:
        arr = arr[-lookback_days:]

    df = pd.DataFrame(arr[:, :6], columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["ts_ms"] = df["ts_ms"].astype("int64")
    df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)

    # Минимум данных: нужно чтобы посчитались роллинги
    min_need = max(int(model.feature_cfg.win_sma_slow), int(model.feature_cfg.win_vol), int(model.feature_cfg.win_sma_fast)) + 5
    if len(df) < min_need:
        raise ValueError(f"Слишком мало свечей для инференса. Нужно хотя бы около {min_need} дневных свечей.")

    df_feat, X_std, feat_cols, _mu, _sd = build_features_1d(df, model.feature_cfg, mu=model.mu, sd=model.sd)

    # инференс
    hmm = model.make_hmm()
    if causal_probs:
        probs = hmm.filter_proba(X_std)
    else:
        probs = hmm.predict_proba_smooth(X_std)

    states = hmm.viterbi(X_std)
    current_probs = probs[-1].copy()
    current_state = int(np.argmax(current_probs))

    out: Dict[str, Any] = {
        "asof_dt": df_feat["dt"].iloc[-1],
        "current_state": current_state,
        "current_state_probs": current_probs,
    }

    if return_history:
        out["dt"] = df_feat["dt"].values
        out["close"] = df_feat["close"].astype(float).values
        out["states"] = states
        out["probs"] = probs

    return out



def current_regime_duration(
    pred_or_states: Union[Dict[str, Any], np.ndarray, list],
    *,
    dt: Optional[Union[np.ndarray, list, pd.Series, pd.DatetimeIndex]] = None,
    current_state: Optional[int] = None,
    use_viterbi: bool = True,
) -> Dict[str, Any]:
    """
    Возвращает, сколько последних свечей подряд держится текущий режим, и когда он начался.

    Поддерживаемые вызовы:

    1) Передать весь результат infer_regime_latest(...):
       dur = current_regime_duration(res, use_viterbi=True)

    2) Передать только states + dt:
       dur = current_regime_duration(res["states"], dt=res["dt"], use_viterbi=True)

    Возврат:
      - current_state: текущий режим (int)
      - bars: сколько последних свечей подряд режим держится (для 1h это часы, для 1d это дни)
      - start_dt: когда начался текущий режим
      - end_dt: дата/время последней свечи в окне
      - duration_time: длительность по времени (end - start + один шаг свечи)
    """
    # --- распаковка входа ---
    if isinstance(pred_or_states, dict):
        pred = pred_or_states
        if "states" not in pred or "dt" not in pred:
            raise ValueError("Нужно, чтобы в pred были 'dt' и 'states' (return_history=True или дефолт вашей версии).")

        states_arr = np.asarray(pred["states"], dtype=int)

        dt_raw = pred["dt"]
        dt_idx = pd.DatetimeIndex(pd.to_datetime(dt_raw, utc=True))

        pred_current_state = pred.get("current_state", None)
        pred_current_state = int(pred_current_state) if pred_current_state is not None else None

    else:
        states_arr = np.asarray(pred_or_states, dtype=int)
        if dt is None:
            raise ValueError("Если вы передаёте только states, нужно ещё передать dt=... (того же размера).")

        dt_idx = pd.DatetimeIndex(pd.to_datetime(dt, utc=True))

        pred_current_state = int(current_state) if current_state is not None else None

    # --- проверки ---
    if states_arr.ndim != 1:
        states_arr = states_arr.reshape(-1)
    if len(states_arr) == 0:
        raise ValueError("Пустой массив states.")
    if len(dt_idx) != len(states_arr):
        raise ValueError(f"dt и states должны быть одинаковой длины: dt={len(dt_idx)} states={len(states_arr)}")

    # --- текущий режим ---
    if use_viterbi:
        cur_state = int(states_arr[-1])
    else:
        if pred_current_state is None:
            raise ValueError("use_viterbi=False, но current_state не передан и не найден в pred.")
        cur_state = int(pred_current_state)

    # --- считаем длину текущего сегмента с конца ---
    i = len(states_arr) - 1
    while i >= 0 and int(states_arr[i]) == cur_state:
        i -= 1

    start_idx = i + 1
    bars = len(states_arr) - start_idx

    start_dt = dt_idx[start_idx]
    end_dt = dt_idx[-1]

    # шаг свечи по двум последним точкам времени (если есть)
    if len(dt_idx) >= 2:
        bar_step = dt_idx[-1] - dt_idx[-2]
    else:
        bar_step = pd.Timedelta(0)

    duration_time = (end_dt - start_dt) + bar_step

    return {
        "current_state": cur_state,
        "bars": int(bars),
        "start_dt": start_dt,
        "end_dt": end_dt,
        "duration_time": duration_time,
    }

# =============================================================================
# Example usage (без CLI)
# =============================================================================
# if __name__ == "__main__":
#     # Пример, как вызвать обучение вручную:
#     # 1) Поменяйте пути на свои.
#     CSV = "BTCUSDT_1d.csv"
#     MODEL_PATH = "hmm_model_btc_1d.npz"
#     OUT = "hmm_artifacts"

#     if os.path.exists(CSV):
#         artifacts = train_hmm_regime_model(
#             csv_path=CSV,
#             model_path=MODEL_PATH,
#             out_dir=OUT,
#             start="2020-01-01",
#             end=None,
#             n_states=4,
#             n_iter=200,
#             tol=1e-4,
#             seed=42,
#             has_header=False,
#             feature_cfg=FeatureConfig(win_vol=20, win_sma_fast=20, win_sma_slow=80),
#             init_self_trans=0.95,
#             save_zoom_last_year=True,
#         )
#         print("Saved artifacts:")
#         for k, v in artifacts.items():
#             if v:
#                 print(f"  {k}: {v}")

#         m = load_hmm_regime_model(MODEL_PATH)

#         # Пример прод-инференса: возьмём последние 200 свечей из CSV
#         df0 = read_ohlcv_csv(CSV, has_header=False)
#         last = df0[["ts_ms", "open", "high", "low", "close", "volume"]].values[-200:]
#         pred = infer_regime_latest(last, m, lookback_days=180, return_history=False, causal_probs=True)
#         print("\nProd inference:")
#         print("  asof_dt:", pred["asof_dt"])
#         print("  current_state:", pred["current_state"])
#         print("  current_state_probs:", pred["current_state_probs"])
