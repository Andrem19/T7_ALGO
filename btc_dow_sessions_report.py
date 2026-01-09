# debug_long_portability.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
VECTOR_CSV = "/home/jupiter/PYTHON/T7_ALGO/vector.csv"  # поменяй под себя

DOW_ALLOW = (0, 1, 2, 3, 6)          # Mon Tue Wed Thu Sun
IV_SET = (3, 4)
RSI_SET = (1,)
CL1H_SET = (2,)

# какие варианты часов хотим сравнить
HOURS_VARIANTS = {
    "vote_h6": (6,),
    "vote_h456": (4, 5, 6),
    "vote_h345": (3, 4, 5),
}

# strict-окна для сравнения
STRICT_VARIANTS = {
    "strict_345": (3, 4, 5),
    "strict_456": (4, 5, 6),
}


# -----------------------------
# HELPERS
# -----------------------------
def _stats(p: np.ndarray) -> dict:
    p = np.asarray(p, dtype="float64")
    p = p[np.isfinite(p)]
    if p.size == 0:
        return {"n": 0, "mean": np.nan, "win_rate": np.nan, "p05": np.nan, "min": np.nan}
    return {
        "n": int(p.size),
        "mean": float(p.mean()),
        "win_rate": float((p > 0).mean()),
        "p05": float(np.quantile(p, 0.05)),
        "min": float(p.min()),
    }


def load_vector(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "tm_ms" not in df.columns:
        df = df.rename(columns={df.columns[0]: "tm_ms"})

    df["tm_ms"] = pd.to_numeric(df["tm_ms"], errors="coerce")
    df = df.dropna(subset=["tm_ms"]).copy()
    df["tm_ms"] = df["tm_ms"].astype("int64")

    df["dt"] = pd.to_datetime(df["tm_ms"], unit="ms", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)

    df["dow"] = df["dt"].dt.dayofweek
    df["hour"] = df["dt"].dt.hour
    df["year"] = df["dt"].dt.year

    for c in ("profit_1", "iv_est", "rsi", "cl_1h"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["profit_1", "iv_est", "rsi", "cl_1h", "dow", "hour", "year"]).copy()
    return df


def mask_vote(df: pd.DataFrame, hours_vote: tuple[int, ...], k: int = 3) -> pd.Series:
    m_dow = df["dow"].isin(DOW_ALLOW)
    c_iv = df["iv_est"].isin(IV_SET)
    c_rsi = df["rsi"].isin(RSI_SET)
    c_h = df["hour"].isin(hours_vote)
    c_cl = df["cl_1h"].isin(CL1H_SET)

    votes = c_iv.astype(int) + c_rsi.astype(int) + c_h.astype(int) + c_cl.astype(int)
    return m_dow & (votes >= k)


def mask_strict(df: pd.DataFrame, hours: tuple[int, ...], need_at_least: int = 2) -> pd.Series:
    m_dow = df["dow"].isin(DOW_ALLOW)
    c_iv = df["iv_est"].isin(IV_SET)
    c_rsi = df["rsi"].isin(RSI_SET)
    c_cl = df["cl_1h"].isin(CL1H_SET)

    score3 = c_iv.astype(int) + c_rsi.astype(int) + c_cl.astype(int)
    return m_dow & df["hour"].isin(hours) & (score3 >= need_at_least)


def decompose_core_relax(df: pd.DataFrame, hours_relax: tuple[int, ...]) -> tuple[pd.Series, pd.Series]:
    """
    CORE: iv & rsi & cl (в любой час, но только на разрешённых DOW)
    RELAX: в hours_relax и ровно 2 из {iv,rsi,cl}
    """
    m_dow = df["dow"].isin(DOW_ALLOW)
    c_iv = df["iv_est"].isin(IV_SET)
    c_rsi = df["rsi"].isin(RSI_SET)
    c_cl = df["cl_1h"].isin(CL1H_SET)
    score3 = c_iv.astype(int) + c_rsi.astype(int) + c_cl.astype(int)

    core = m_dow & (score3 == 3)
    relax = m_dow & df["hour"].isin(hours_relax) & (score3 == 2)
    return core, relax


def report_by_year(df: pd.DataFrame, mask: pd.Series, title: str) -> None:
    years = sorted(df["year"].unique())
    rows = []
    for y in years:
        p = df.loc[mask & (df["year"] == y), "profit_1"].to_numpy()
        s = _stats(p)
        rows.append({"year": int(y), **s})
    out = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(title)
    print(out.to_string(index=False))


def main() -> None:
    df = load_vector(VECTOR_CSV)

    # 1) Сравнение vote-вариантов
    for name, hs in HOURS_VARIANTS.items():
        m = mask_vote(df, hs, k=3)
        report_by_year(df, m, f"[VOTE] {name} hours={hs}  (k>=3 of iv,rsi,h,cl)")

        core, relax = decompose_core_relax(df, hs)
        report_by_year(df, core, f"    CORE (iv&rsi&cl) for {name}")
        report_by_year(df, relax, f"    RELAX (hours in {hs} and exactly 2 of iv,rsi,cl) for {name}")

    # 2) Сравнение strict-вариантов
    for name, hs in STRICT_VARIANTS.items():
        m = mask_strict(df, hs, need_at_least=2)
        report_by_year(df, m, f"[STRICT] {name} hours={hs}  (only hours & >=2 of iv,rsi,cl)")

    print("\nDONE.")


if __name__ == "__main__":
    main()
