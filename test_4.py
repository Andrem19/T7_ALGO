# rule_portability_timeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from helpers.mprint import green
import os
import shared_vars as sv
from helpers.get_data import load_data_sets
from itertools import combinations


import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------
# Defaults
# -----------------------------
HOURS = list(range(24))  # 0..23
DEFAULT_TIMELINE_BINS = 24
DEFAULT_TOP_FRAC = 0.2
DEFAULT_MIN_BASE_ROWS_PER_BIN = 3

best = {
    'rule': None,
    'value': 0
}

# -----------------------------
# Rule parsing
# -----------------------------

_CMP_RE = re.compile(
    r"^\s*(<=|>=|<|>|==|=|!=)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)


@dataclass(frozen=True)
class ParsedRule:
    col: str
    kind: str  # "in" | "cmp"
    values: Optional[Tuple[Any, ...]] = None
    op: Optional[str] = None
    thr: Optional[float] = None
    text: str = ""


def _parse_rule_spec(col: str, spec: Any) -> ParsedRule:
    """
    Supported specs:
      - list/tuple/set: [0,1,2]  =>  col in (0,1,2)
      - string comparator: '>0.33', '<=76', '!=0', '==10'
    """
    if not isinstance(col, str) or not col.strip():
        raise ValueError("Rule column name must be a non-empty string.")

    # "in" rule
    if isinstance(spec, (list, tuple, set)):
        vals = tuple(spec)
        if len(vals) == 0:
            raise ValueError(f"Rule for '{col}' has empty list of allowed values.")
        return ParsedRule(
            col=col,
            kind="in",
            values=vals,
            text=f"{col} in {vals}",
        )

    # comparator rule in string
    if isinstance(spec, str):
        m = _CMP_RE.match(spec)
        if not m:
            raise ValueError(
                f"Rule for '{col}' is a string but not a comparator. "
                f"Expected like '>0.33', '<=76', '!=0', '==10'. Got: {spec!r}"
            )
        op = m.group(1)
        if op == "=":
            op = "=="
        thr = float(m.group(2))
        return ParsedRule(
            col=col,
            kind="cmp",
            op=op,
            thr=thr,
            text=f"{col} {op} {thr}",
        )

    raise ValueError(
        f"Unsupported rule spec type for '{col}': {type(spec)}. "
        "Use list/tuple/set for membership or string comparator like '>0.33'."
    )


def _parse_rules(rules: Dict[str, Any]) -> List[ParsedRule]:
    if not isinstance(rules, dict) or not rules:
        raise ValueError("rules must be a non-empty dict like {'atr':[1,2], 'rsi':'>31'}")

    parsed: List[ParsedRule] = []
    for col, spec in rules.items():
        parsed.append(_parse_rule_spec(col, spec))
    return parsed


def _make_rule_mask(df: pd.DataFrame, rule: ParsedRule) -> np.ndarray:
    if rule.col not in df.columns:
        raise ValueError(f"Rule column '{rule.col}' not found in CSV columns.")

    if rule.kind == "in":
        allowed = set(rule.values or ())
        # membership on original dtype (fast and safe)
        return df[rule.col].isin(allowed).to_numpy(dtype=np.bool_, copy=False)

    if rule.kind == "cmp":
        s = pd.to_numeric(df[rule.col], errors="coerce")
        x = s.to_numpy(dtype=np.float64, copy=False)
        thr = float(rule.thr)

        if rule.op == ">":
            return (x > thr) & np.isfinite(x)
        if rule.op == ">=":
            return (x >= thr) & np.isfinite(x)
        if rule.op == "<":
            return (x < thr) & np.isfinite(x)
        if rule.op == "<=":
            return (x <= thr) & np.isfinite(x)
        if rule.op == "==":
            return (x == thr) & np.isfinite(x)
        if rule.op == "!=":
            return (x != thr) & np.isfinite(x)

        raise ValueError(f"Unsupported comparator op: {rule.op!r}")

    raise ValueError(f"Unsupported rule kind: {rule.kind!r}")


# -----------------------------
# Metrics result
# -----------------------------

@dataclass(frozen=True)
class RulePortabilityResult:
    rule_col: str
    rule_text: str

    base_rows: int
    base_profit_sum: float
    base_profit_mean: float

    without_rows: int
    without_profit_sum: float
    without_profit_mean: float

    mean_degradation_global: float

    bins: int
    base_bins_covered: int
    base_bins_coverage: float

    bins_pos: int
    bins_neg: int
    bins_zero: int

    evenness: float
    top_share: float
    portability_score: float

    worst_bins_pos: Tuple[Tuple[int, float], ...]
    worst_bins_neg: Tuple[Tuple[int, float], ...]


# -----------------------------
# Helpers
# -----------------------------

def _make_bin_id(
    df: pd.DataFrame,
    *,
    bins: int,
    sort_by_tm_ms: bool,
    bin_mode: str,
    tm_col: str,
) -> np.ndarray:
    """
    bin_mode:
      - "time:  equal time span chunks using tm_ms
      - "rows":  equal row-count chunks by index
    """
    if bins < 2:
        raise ValueError("bins must be >= 2.")
    if bin_mode not in ("time", "rows"):
        raise ValueError("bin_mode must be 'time' or 'rows'.")

    if sort_by_tm_ms and tm_col in df.columns:
        df.sort_values(tm_col, kind="mergesort", inplace=True, ignore_index=True)

    n = int(len(df))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    if bin_mode == "rows" or tm_col not in df.columns:
        idx = np.arange(n, dtype=np.int64)
        b = (idx * int(bins) // max(1, n)).astype(np.int64, copy=False)
        return np.minimum(b, int(bins) - 1)

    tm = pd.to_numeric(df[tm_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if np.isnan(tm).any():
        idx = np.arange(n, dtype=np.float64)
        tm = np.where(np.isnan(tm), idx, tm)

    t_min = float(np.min(tm))
    t_max = float(np.max(tm))
    if t_max <= t_min:
        return np.zeros(n, dtype=np.int64)

    scaled = (tm - t_min) / (t_max - t_min)
    b = np.floor(scaled * float(bins)).astype(np.int64, copy=False)
    return np.clip(b, 0, int(bins) - 1)


def _safe_mean(sum_: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    out = np.full_like(sum_, fill_value=np.nan, dtype=np.float64)
    ok = cnt > 0
    out[ok] = sum_[ok] / cnt[ok]
    return out


def _entropy_evenness(values: np.ndarray) -> float:
    """
    values: non-negative by bins (e.g., positive degradation per bin).
    Returns 0..1. Closer to 1 => more evenly spread.
    """
    v = values[np.isfinite(values)]
    v = v[v > 0]
    if v.size <= 1:
        return 0.0

    s = float(v.sum())
    if s <= 0:
        return 0.0

    p = v / s
    ent = float(-(p * np.log(p)).sum())
    ent_max = float(np.log(v.size))
    return float(ent / ent_max) if ent_max > 0 else 0.0


def _top_share(values: np.ndarray, *, top_frac: float) -> float:
    v = values[np.isfinite(values)]
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    v_sorted = np.sort(v)[::-1]
    top_k = int(max(1, math.ceil(float(top_frac) * float(v_sorted.size))))
    total = float(v_sorted.sum())
    if total <= 0:
        return 0.0
    return float(v_sorted[:top_k].sum() / total)


def _bincount_sum_and_count(
    bin_id: np.ndarray,
    profit: np.ndarray,
    mask: np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    b = bin_id[mask]
    p = profit[mask]
    cnt = np.bincount(b, minlength=bins).astype(np.int64, copy=False)
    s = np.bincount(b, weights=p, minlength=bins).astype(np.float64, copy=False)
    return s, cnt


def _build_prefix_suffix(masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    prefix[i]  = AND masks[0..i-1]
    suffix[i]  = AND masks[i..end-1]
    all-but-i = prefix[i] & suffix[i+1]
    """
    n = len(masks)
    if n == 0:
        return [np.array([], dtype=np.bool_)], [np.array([], dtype=np.bool_)]

    prefix: List[np.ndarray] = [np.ones_like(masks[0], dtype=np.bool_)]
    for i in range(n):
        prefix.append(prefix[-1] & masks[i])

    suffix: List[np.ndarray] = [np.ones_like(masks[0], dtype=np.bool_) for _ in range(n + 1)]
    suffix[n] = np.ones_like(masks[0], dtype=np.bool_)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] & masks[i]

    return prefix, suffix


def _compute_bin_periods(
    df: pd.DataFrame,
    bin_id: np.ndarray,
    *,
    bins: int,
    tm_col: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if tm_col not in df.columns:
        for b in range(int(bins)):
            out.append({"bin": b, "rows": int(np.sum(bin_id == b)), "dt0_utc": None, "dt1_utc": None})
        return out

    tm_ms = pd.to_numeric(df[tm_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(tm_ms)

    for b in range(int(bins)):
        idx = (bin_id == b) & ok
        n_rows_bin = int(idx.sum())
        if n_rows_bin <= 0:
            out.append({"bin": b, "rows": 0, "dt0_utc": None, "dt1_utc": None})
            continue

        t0 = float(np.min(tm_ms[idx]))
        t1 = float(np.max(tm_ms[idx]))

        dt0 = pd.to_datetime(int(t0), unit="ms", utc=True)
        dt1 = pd.to_datetime(int(t1), unit="ms", utc=True)

        out.append({"bin": b, "rows": n_rows_bin, "dt0_utc": dt0, "dt1_utc": dt1})
    return out


def _prepare_price_series_from_ohlcv_1d(data_1d: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    data_1d: [[timestamp_ms, open, high, low, close, volume], ...]
    Returns (dt_utc, close) or None.
    """
    if data_1d is None:
        return None
    arr = np.asarray(data_1d)
    if arr.ndim != 2 or arr.shape[1] < 5:
        return None

    tm = arr[:, 0].astype(np.float64, copy=False)
    cl = arr[:, 4].astype(np.float64, copy=False)

    ok = np.isfinite(tm) & np.isfinite(cl)
    if not ok.any():
        return None

    tm = tm[ok]
    cl = cl[ok]

    order = np.argsort(tm)
    tm = tm[order]
    cl = cl[order]

    dt = pd.to_datetime(tm.astype(np.int64, copy=False), unit="ms", utc=True).to_numpy()
    return dt, cl


def _bin_time_geometry(bin_periods: List[Dict[str, Any]], bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      x_left: left edge per bin (matplotlib date number) or nan
      x_mid:  middle per bin (matplotlib date number) or nan
      width:  width per bin (in matplotlib date units) or nan
    """
    import matplotlib.dates as mdates

    x_left = np.full(int(bins), np.nan, dtype=np.float64)
    x_mid = np.full(int(bins), np.nan, dtype=np.float64)
    width = np.full(int(bins), np.nan, dtype=np.float64)

    for b in range(int(bins)):
        dt0 = bin_periods[b].get("dt0_utc")
        dt1 = bin_periods[b].get("dt1_utc")
        if dt0 is None or dt1 is None:
            continue

        n0 = mdates.date2num(dt0.to_pydatetime())
        n1 = mdates.date2num(dt1.to_pydatetime())
        if n1 <= n0:
            n1 = n0 + 1e-6

        x_left[b] = n0
        width[b] = n1 - n0
        x_mid[b] = (n0 + n1) / 2.0

    return x_left, x_mid, width


def _plot_all_rules_stacked(
    *,
    profit_col: str,
    bins: int,
    bin_periods: List[Dict[str, Any]],
    results_sorted: List[RulePortabilityResult],
    diagnostics: Dict[str, Dict[str, Any]],
    price_series: Optional[Tuple[np.ndarray, np.ndarray]],
    title_prefix: str,
    show: bool,
    save_path: Optional[str],
    dpi: int,
    shade_non_considered: bool,
) -> None:
    """
    One figure:
      - top: price (close)
      - below: one row per rule with degradation bars by bins
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    n_rules = int(len(results_sorted))
    if n_rules <= 0:
        return

    x_left, x_mid, width = _bin_time_geometry(bin_periods, bins)

    # common y-scale for honest comparison
    global_max = 0.0
    for r in results_sorted:
        d = diagnostics[r.rule_col]
        degr = np.asarray(d["degr"], dtype=np.float64)
        considered = np.asarray(d["considered"], dtype=np.bool_)
        mask = np.isfinite(degr) & considered & np.isfinite(x_mid)
        if mask.any():
            m = float(np.nanmax(np.abs(degr[mask])))
            if m > global_max:
                global_max = m
    if not np.isfinite(global_max) or global_max <= 0.0:
        global_max = 1.0
    y_lim = global_max * 1.10

    fig_h = 3.2 + 1.05 * n_rules
    fig, axes = plt.subplots(
        1 + n_rules,
        1,
        figsize=(16, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4] + [1.0] * n_rules},
    )

    ax_price = axes[0]
    rule_axes = axes[1:]

    # price
    if price_series is not None:
        dt, close = price_series
        ax_price.plot(dt, close)
        ax_price.set_ylabel("Close (data_1d)")
        ax_price.grid(True)
    else:
        ax_price.text(0.02, 0.5, "No price series (data_1d is None or invalid)", transform=ax_price.transAxes)
        ax_price.set_ylabel("Price")
        ax_price.grid(True)

    # bin boundaries on all axes
    for b in range(int(bins)):
        if not np.isfinite(x_left[b]):
            continue
        for ax in axes:
            ax.axvline(x_left[b], linewidth=0.7)

    # rules
    for ax, r in zip(rule_axes, results_sorted):
        d = diagnostics[r.rule_col]
        degr = np.asarray(d["degr"], dtype=np.float64)
        considered = np.asarray(d["considered"], dtype=np.bool_)

        y = np.full(int(bins), np.nan, dtype=np.float64)
        ok = np.isfinite(x_mid) & np.isfinite(width) & considered & np.isfinite(degr)
        y[ok] = degr[ok]

        if shade_non_considered:
            for b in range(int(bins)):
                if not np.isfinite(x_left[b]) or not np.isfinite(width[b]):
                    continue
                if not considered[b]:
                    ax.axvspan(x_left[b], x_left[b] + width[b], alpha=0.12)

        ax.bar(x_mid[ok], y[ok], width=width[ok], align="center")
        ax.axhline(0.0, linewidth=1.0)
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(True)

        info = (
            f"{r.rule_text} | score={r.portability_score:.3f} | "
            f"glob_degr={r.mean_degradation_global:.3f} | even={r.evenness:.3f} | top={r.top_share:.3f}"
        )
        ax.text(0.01, 0.78, info, transform=ax.transAxes)

    # x formatting
    ax_last = rule_axes[-1]
    ax_last.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_last.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_last.xaxis.get_major_locator()))

    fig.suptitle(f"{title_prefix} | {profit_col} | bins={bins}", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=int(dpi))

    if show:
        plt.show()
    else:
        plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def evaluate_rules_portability_timeline(
    csv_path: str,
    data_1d: Optional[np.ndarray],
    rules: Dict[str, Any],
    *,
    profit_col: str = "profit_2",
    bins: int = DEFAULT_TIMELINE_BINS,
    bin_mode: str = "time",
    sort_by_tm_ms: bool = True,
    tm_col: str = "tm_ms",
    min_base_rows_per_bin: int = DEFAULT_MIN_BASE_ROWS_PER_BIN,
    top_frac: float = DEFAULT_TOP_FRAC,
    print_results: bool = True,

    # visual
    plot_stacked: bool = True,
    show_plots: bool = True,
    save_plot_path: Optional[str] = None,
    plot_dpi: int = 150,
    shade_non_considered_bins: bool = True,

    # if many rules: show only top N by portability_score
    max_rules_to_plot: Optional[int] = None,
) -> Dict[str, Any]:
    """
    rules supports:
      {'atr':[0,3,4], 'rsi':'>31', 'iv_est':'>0.33', ...}
      {'rsi':'<76', ...}

    We compute:
      baseline = ALL rules applied
      for each rule: remove that rule only, keep the rest, compute degradation per bin:
        degradation(bin) = baseline_mean(bin) - mean_without_rule(bin)
    """
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError("top_frac must be in (0, 1].")
    if min_base_rows_per_bin < 1:
        raise ValueError("min_base_rows_per_bin must be >= 1.")
    if bins < 2:
        raise ValueError("bins must be >= 2.")

    parsed_rules = _parse_rules(rules)

    df = pd.read_csv(csv_path)
    if profit_col not in df.columns:
        raise ValueError(f"profit_col '{profit_col}' not found in CSV columns.")

    # binning (may sort df in place)
    bin_id = _make_bin_id(df, bins=bins, sort_by_tm_ms=sort_by_tm_ms, bin_mode=bin_mode, tm_col=tm_col)
    bin_periods = _compute_bin_periods(df, bin_id, bins=bins, tm_col=tm_col)

    if print_results:
        print("=" * 110)
        print(f"BIN PERIODS by '{tm_col}' (UTC):")
        for bp in bin_periods:
            b = bp["bin"]
            rows = bp["rows"]
            dt0 = bp.get("dt0_utc")
            dt1 = bp.get("dt1_utc")
            if dt0 is None or dt1 is None:
                print(f"  bin {b:02d}: <empty> | rows={rows}")
            else:
                print(f"  bin {b:02d}: {dt0} .. {dt1} | rows={rows}")
        print("=" * 110)

    # profit vector
    profit = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)

    # rule masks
    rule_masks: List[np.ndarray] = []
    rule_cols: List[str] = []
    rule_texts: Dict[str, str] = {}

    for r in parsed_rules:
        rule_cols.append(r.col)
        rule_texts[r.col] = r.text
        rule_masks.append(_make_rule_mask(df, r))

    # prefix/suffix
    prefix, suffix = _build_prefix_suffix(rule_masks)

    # baseline
    mask_all = prefix[len(rule_masks)]
    if not mask_all.any():
        raise ValueError("Baseline mask (all rules applied) selects 0 rows. Rules are too strict or mismatch.")

    base_sum_bins, base_cnt_bins = _bincount_sum_and_count(bin_id, profit, mask_all, bins=bins)
    base_mean_bins = _safe_mean(base_sum_bins, base_cnt_bins)

    considered = base_cnt_bins >= int(min_base_rows_per_bin)
    base_bins_covered = int(considered.sum())
    base_bins_coverage = float(base_bins_covered / bins) if bins > 0 else 0.0

    base_rows = int(mask_all.sum())
    base_profit_sum = float(profit[mask_all].sum())
    base_profit_mean = float(base_profit_sum / base_rows) if base_rows > 0 else float("nan")

    price_series = _prepare_price_series_from_ohlcv_1d(data_1d)

    per_rule_results: List[RulePortabilityResult] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}

    for i, col in enumerate(rule_cols):
        # remove this rule only
        mask_wo = prefix[i] & suffix[i + 1]
        if not mask_wo.any():
            mask_wo = np.zeros_like(mask_all, dtype=np.bool_)

        wo_sum_bins, wo_cnt_bins = _bincount_sum_and_count(bin_id, profit, mask_wo, bins=bins)
        wo_mean_bins = _safe_mean(wo_sum_bins, wo_cnt_bins)

        degr = np.full(int(bins), np.nan, dtype=np.float64)
        degr[considered] = base_mean_bins[considered] - wo_mean_bins[considered]

        bins_pos = int(np.sum((degr > 0) & np.isfinite(degr)))
        bins_neg = int(np.sum((degr < 0) & np.isfinite(degr)))
        bins_zero = int(np.sum((degr == 0) & np.isfinite(degr)))

        degr_pos = np.zeros(int(bins), dtype=np.float64)
        ok_pos = (degr > 0) & np.isfinite(degr)
        degr_pos[ok_pos] = degr[ok_pos]

        evenness = _entropy_evenness(degr_pos)
        top_share = _top_share(degr_pos, top_frac=float(top_frac))
        portability_score = float(base_bins_coverage * evenness * (1.0 - top_share))

        wo_rows = int(mask_wo.sum())
        wo_profit_sum = float(profit[mask_wo].sum()) if wo_rows > 0 else 0.0
        wo_profit_mean = float(wo_profit_sum / wo_rows) if wo_rows > 0 else float("nan")

        mean_degradation_global = float(base_profit_mean - wo_profit_mean)

        pos_items = [(int(b), float(degr[b])) for b in range(int(bins)) if np.isfinite(degr[b]) and degr[b] > 0]
        neg_items = [(int(b), float(degr[b])) for b in range(int(bins)) if np.isfinite(degr[b]) and degr[b] < 0]
        pos_items.sort(key=lambda x: x[1], reverse=True)
        neg_items.sort(key=lambda x: x[1])

        res = RulePortabilityResult(
            rule_col=col,
            rule_text=rule_texts.get(col, col),

            base_rows=base_rows,
            base_profit_sum=base_profit_sum,
            base_profit_mean=base_profit_mean,

            without_rows=wo_rows,
            without_profit_sum=wo_profit_sum,
            without_profit_mean=wo_profit_mean,

            mean_degradation_global=mean_degradation_global,

            bins=int(bins),
            base_bins_covered=base_bins_covered,
            base_bins_coverage=base_bins_coverage,

            bins_pos=bins_pos,
            bins_neg=bins_neg,
            bins_zero=bins_zero,

            evenness=float(evenness),
            top_share=float(top_share),
            portability_score=float(portability_score),

            worst_bins_pos=tuple(pos_items[:5]),
            worst_bins_neg=tuple(neg_items[:5]),
        )
        per_rule_results.append(res)

        diagnostics[col] = {
            "degr": degr,
            "considered": considered,
        }

    # sort by portability_score desc
    per_rule_results.sort(key=lambda r: r.portability_score, reverse=True)

    out = {
        "csv_path": csv_path,
        "profit_col": profit_col,
        "bins": int(bins),
        "bin_mode": bin_mode,
        "min_base_rows_per_bin": int(min_base_rows_per_bin),
        "top_frac": float(top_frac),
        "baseline": {
            "rows": base_rows,
            "profit_sum": base_profit_sum,
            "profit_mean": base_profit_mean,
            "bins_coverage": base_bins_coverage,
            "bins_covered": base_bins_covered,
        },
        "bin_periods": bin_periods,
        "results": per_rule_results,
    }

    if print_results:
        print("=" * 110)
        print(f"PORTABILITY CHECK | profit_col={profit_col} | bins={bins} | bin_mode={bin_mode} | top_frac={top_frac}")
        print(f"Baseline: rows={base_rows} | profit_sum={base_profit_sum:.6f} | mean={base_profit_mean:.6f}")
        print(f"Baseline bins coverage: {base_bins_covered}/{bins} = {base_bins_coverage:.3f} "
              f"(min_rows_per_bin={min_base_rows_per_bin})")
        print("-" * 110)
        print("RULES (parsed):")
        for r in parsed_rules:
            print(f"  - {r.text}")
        print("-" * 110)
        for r in per_rule_results:
            if r.mean_degradation_global > best['value']:
                best['value'] = r.mean_degradation_global
                best['rule'] = r.rule_text
        for r in per_rule_results:
            print(f"RULE: {r.rule_text}")
            print(f"  Global mean degradation (remove rule -> mean gets worse by): {r.mean_degradation_global:.6f}")
            print(f"  Without rule: rows={r.without_rows} | mean={r.without_profit_mean:.6f} | sum={r.without_profit_sum:.6f}")
            print(f"  Timeline: coverage={r.base_bins_coverage:.3f} | pos_bins={r.bins_pos} | neg_bins={r.bins_neg} | zero_bins={r.bins_zero}")
            print(f"  Uniformity: evenness={r.evenness:.3f} | top_share={r.top_share:.3f} | portability_score={r.portability_score:.3f}")
            if r.worst_bins_pos:
                print(f"  Worst POS bins (remove rule hurts most): {r.worst_bins_pos}")
            if r.worst_bins_neg:
                print(f"  Worst NEG bins (remove rule helps): {r.worst_bins_neg}")
            print("-" * 110)

    # plot: stacked
    if plot_stacked:
        results_for_plot = per_rule_results
        if max_rules_to_plot is not None:
            results_for_plot = results_for_plot[: int(max_rules_to_plot)]

        title_prefix = f"Rules portability (stacked): {len(results_for_plot)}/{len(per_rule_results)} rules shown"
        _plot_all_rules_stacked(
            profit_col=profit_col,
            bins=int(bins),
            bin_periods=bin_periods,
            results_sorted=results_for_plot,
            diagnostics=diagnostics,
            price_series=price_series,
            title_prefix=title_prefix,
            show=bool(show_plots),
            save_path=save_plot_path,
            dpi=int(plot_dpi),
            shade_non_considered=bool(shade_non_considered_bins),
        )

    return out


# -----------------------------
# Example usage
# -----------------------------

def example_run() -> None:
    # Example: replace with your own data loader
    # data_1d must be ndarray like [[timestamp_ms, open, high, low, close, volume], ...]
    sv.data_1d = load_data_sets(1440)

    rules_1 = {
        'iv_est': [3,4],
        'rsi': [1],
        # 'd': [0,1,2,5,6,7],
        # 'd': [4],
        # 'cl_1h': [2],
        # 'cl_1h': [2],
        # 'h': [6]
        # 'cl_1d': [0,1,2,4]
        'squize_index': [2,3]
    }

    rules_2 = {
        # 'd': [4],
        # 'cl_15m': [3],
        # 'cl_1d': [2]
        # 'cl_4h': [2]
        # 'h': [15]

        # 'squize_index': [1,3,4]
    }
    

           
    evaluate_rules_portability_timeline(
        csv_path="vector.csv",
        data_1d=sv.data_1d,
        rules=rules_1,
        profit_col="profit_1",
        bins=24,
        bin_mode="time",
        sort_by_tm_ms=True,
        min_base_rows_per_bin=3,
        top_frac=0.2,
        print_results=True,

        plot_stacked=True,
        show_plots=True,
        save_plot_path=None,   # e.g. "plots/portability_stacked_profit2.png"
        plot_dpi=160,
        shade_non_considered_bins=True,
        max_rules_to_plot=None,
    )

            


if __name__ == "__main__":
    example_run()
