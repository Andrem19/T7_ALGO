# btc_dow_sessions_report.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Чтобы работало на сервере/без X-сервера
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------
# Конфигурация (можно менять)
# -----------------------------
# Сессии в UTC: Asia 00–07, EU 08–15, US 16–23
SESSIONS_UTC: Dict[str, Tuple[int, int]] = {
    "Asia": (0, 7),
    "EU": (8, 15),
    "US": (16, 23),
}

OUT_DIR_DEFAULT = "./btc_dow_report"
PLOT_DPI_DEFAULT = 160


# -----------------------------
# Вспомогательное
# -----------------------------
def _looks_like_int_token(x: str) -> bool:
    s = str(x).strip().strip('"').strip("'")
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return s.isdigit()


def _detect_header_first_cell(csv_path: str) -> bool:
    """
    Возвращает True, если первая строка похожа на заголовок (а не на данные).
    Логика: если первая ячейка не похожа на целое число (таймстамп), значит это header.
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip()
    if not first_line:
        return True
    first_cell = first_line.split(",")[0].strip()
    return not _looks_like_int_token(first_cell)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_timestamp_column(cols: List[str]) -> Optional[str]:
    # приоритеты — самые частые варианты
    candidates = [
        "timestamp_ms",
        "open_time",
        "open_time_ms",
        "timestamp",
        "time",
        "ts",
    ]
    for c in candidates:
        if c in cols:
            return c
    return None


def _pick_datetime_column(cols: List[str]) -> Optional[str]:
    candidates = [
        "datetime_utc",
        "datetime",
        "date",
        "dt",
        "time_utc",
    ]
    for c in candidates:
        if c in cols:
            return c
    return None


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _read_csv_autodetect(csv_path: str) -> pd.DataFrame:
    """
    Читает CSV максимально терпимо:
    - поддерживает наличие/отсутствие хедера
    - поддерживает open_time / timestamp_ms / datetime_utc
    - терпит лишние колонки
    """
    has_header = _detect_header_first_cell(csv_path)

    if has_header:
        df = pd.read_csv(csv_path)
        df = _normalise_columns(df)
    else:
        df = pd.read_csv(csv_path, header=None)
        # Варианты без хедера: минимум первые 6 колонок — это timestamp_ms, open, high, low, close, volume
        n = df.shape[1]
        if n < 5:
            raise ValueError(f"CSV слишком короткий по колонкам: найдено {n}. Нужно минимум 5–6 колонок.")
        # Назначаем имена на первые колонки, остальное игнорируем
        base = ["timestamp_ms", "open", "high", "low", "close", "volume"]
        cols = base[:n]
        df.columns = cols
        df = _normalise_columns(df)

    # Приводим к “основной схеме”: dt (UTC), timestamp_ms, open/high/low/close/volume
    cols = list(df.columns)

    ts_col = _pick_timestamp_column(cols)
    dt_col = _pick_datetime_column(cols)

    if ts_col is None and dt_col is None:
        raise ValueError(
            "Не найдено ни колонки таймстампа, ни колонки даты/времени. "
            f"Доступные колонки: {cols[:30]}"
        )

    # Сначала пробуем dt-колонку (если есть) — она иногда точнее
    dt: Optional[pd.Series] = None
    if dt_col is not None:
        dt = pd.to_datetime(df[dt_col], errors="coerce", utc=True)

    # Теперь таймстамп
    ts_ms: Optional[pd.Series] = None
    if ts_col is not None:
        ts_num = _coerce_numeric_series(df[ts_col])

        if ts_num.notna().any():
            # Определяем: секунды или миллисекунды.
            # Если значения “большие как 13 цифр” — это миллисекунды, иначе секунды.
            med = float(ts_num.dropna().median())
            if med > 1e11:
                # миллисекунды
                ts_ms = ts_num
                if dt is None or dt.isna().all():
                    dt = pd.to_datetime(ts_num.astype("int64", errors="ignore"), unit="ms", utc=True, errors="coerce")
            else:
                # секунды
                ts_ms = ts_num * 1000.0
                if dt is None or dt.isna().all():
                    dt = pd.to_datetime(ts_num.astype("int64", errors="ignore"), unit="s", utc=True, errors="coerce")

    if dt is None:
        raise ValueError("Не удалось собрать временную ось (dt) из CSV.")

    out = pd.DataFrame({"dt": dt})

    # timestamp_ms как int64 (где возможно)
    if ts_ms is None:
        # если таймстампа не было, восстановим из dt
        out["timestamp_ms"] = (out["dt"].view("int64") // 1_000_000).astype("int64")
    else:
        out["timestamp_ms"] = _coerce_numeric_series(ts_ms).round().astype("Int64")

    # Нормализуем OHLCV
    # Иногда Binance экспортирует open/high/low/close как строки — приводим
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            out[c] = _coerce_numeric_series(df[c])
        else:
            out[c] = np.nan

    # Если volume отсутствует (например, некоторые датасеты), заполним нулями
    if out["volume"].isna().all():
        out["volume"] = 0.0

    # Чистим мусор: dt должен быть валидным, close/open не NaN
    out = out.dropna(subset=["dt", "open", "high", "low", "close"]).copy()

    # Сортировка/уникальность
    out = out.sort_values("dt").reset_index(drop=True)
    out = out.drop_duplicates(subset=["dt"], keep="last")

    # Финальные типы
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    return out


def load_ohlcv_1h(csv_path: str) -> pd.DataFrame:
    df = _read_csv_autodetect(csv_path)

    # Добавим удобные поля
    df["date"] = df["dt"].dt.date
    df["dow"] = df["dt"].dt.dayofweek  # Mon=0..Sun=6
    df["hour"] = df["dt"].dt.hour

    # Почасовая доходность: close относительно open
    # Считается так: берём close, делим на open и вычитаем единицу
    df["ret_h"] = (df["close"] / df["open"]) - 1.0

    # Почасовой диапазон: high относительно low
    # Считается так: берём high, делим на low и вычитаем единицу
    df["range_h"] = (df["high"] / df["low"]) - 1.0

    return df


# -----------------------------
# Агрегации по дню/сессиям
# -----------------------------
@dataclass(frozen=True)
class ReportPaths:
    out_dir: str
    img_dir: str


def _ensure_dirs(out_dir: str) -> ReportPaths:
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    return ReportPaths(out_dir=out_dir, img_dir=img_dir)


def _dow_name(d: int) -> str:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if 0 <= d < 7:
        return names[d]
    return str(d)


def _compute_daily(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date", sort=True)

    daily = pd.DataFrame({
        "date": g["dt"].min().dt.date,
        "dt_first": g["dt"].min(),
        "dt_last": g["dt"].max(),
        "open_first": g["open"].first(),
        "close_last": g["close"].last(),
        "high_max": g["high"].max(),
        "low_min": g["low"].min(),
        "dow": g["dow"].first(),
    }).reset_index(drop=True)

    # Дневная доходность: close_last относительно open_first
    daily["day_ret"] = (daily["close_last"] / daily["open_first"]) - 1.0

    # Дневной диапазон: high_max относительно low_min
    daily["day_range"] = (daily["high_max"] / daily["low_min"]) - 1.0

    daily["day_bull"] = daily["day_ret"] > 0.0
    return daily


def _session_slice(df_day: pd.DataFrame, h1: int, h2: int) -> pd.DataFrame:
    # часы включительно: h1..h2
    return df_day[(df_day["hour"] >= h1) & (df_day["hour"] <= h2)]


def _compute_sessions_per_day(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for date, df_day in df.groupby("date", sort=True):
        df_day = df_day.sort_values("dt")
        dow = int(df_day["dow"].iloc[0])

        for sname, (h1, h2) in SESSIONS_UTC.items():
            part = _session_slice(df_day, h1, h2)
            if part.empty:
                continue
            o = float(part["open"].iloc[0])
            c = float(part["close"].iloc[-1])
            ret = (c / o) - 1.0
            rows.append({
                "date": date,
                "dow": dow,
                "session": sname,
                "session_ret": ret,
                "session_bull": ret > 0.0,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Не удалось собрать сессии: проверьте, что данные почасовые и покрывают часы 00..23 UTC.")
    return out.sort_values(["date", "session"]).reset_index(drop=True)


def _agg_daily_by_dow(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("dow", sort=True)
    out = pd.DataFrame({
        "dow": g.size().index.astype(int),
        "days": g.size().values,
        "mean_day_ret_pct": (g["day_ret"].mean().values * 100.0),
        "median_day_ret_pct": (g["day_ret"].median().values * 100.0),
        "winrate_day_pct": (g["day_bull"].mean().values * 100.0),
        "std_day_ret_pct": (g["day_ret"].std(ddof=0).values * 100.0),
        "mean_day_range_pct": (g["day_range"].mean().values * 100.0),
        "median_day_range_pct": (g["day_range"].median().values * 100.0),
    })
    out["dow_name"] = out["dow"].map(_dow_name)
    return out[["dow", "dow_name", "days", "mean_day_ret_pct", "median_day_ret_pct", "winrate_day_pct",
                "std_day_ret_pct", "mean_day_range_pct", "median_day_range_pct"]]


def _agg_sessions_by_dow(sess: pd.DataFrame) -> pd.DataFrame:
    g = sess.groupby(["dow", "session"], sort=True)
    out = g.agg(
        days=("session_ret", "size"),
        mean_session_ret_pct=("session_ret", lambda x: float(x.mean() * 100.0)),
        median_session_ret_pct=("session_ret", lambda x: float(x.median() * 100.0)),
        winrate_session_pct=("session_bull", lambda x: float(x.mean() * 100.0)),
        std_session_ret_pct=("session_ret", lambda x: float(x.std(ddof=0) * 100.0)),
    ).reset_index()
    out["dow_name"] = out["dow"].map(_dow_name)
    return out[["dow", "dow_name", "session", "days", "mean_session_ret_pct", "median_session_ret_pct",
                "winrate_session_pct", "std_session_ret_pct"]]


def _session_conditionals_by_dow(sess: pd.DataFrame) -> pd.DataFrame:
    """
    Считаем вероятности:
    - EU bull | Asia bull / Asia bear
    - US bull | EU bull / EU bear
    отдельно по каждому дню недели.
    """
    out_rows: List[Dict[str, Any]] = []

    # pivot day-session into columns
    piv = sess.pivot_table(index=["date", "dow"], columns="session", values="session_bull", aggfunc="last")
    piv = piv.reset_index()

    for dow, part in piv.groupby("dow", sort=True):
        def _prob(cond_mask: pd.Series, target_col: str) -> Tuple[float, int]:
            sub = part.loc[cond_mask, target_col]
            sub = sub.dropna()
            if len(sub) == 0:
                return (np.nan, 0)
            return (float(sub.mean() * 100.0), int(len(sub)))

        # EU | Asia
        if "Asia" in part.columns and "EU" in part.columns:
            p1, n1 = _prob(part["Asia"] == True, "EU")
            p2, n2 = _prob(part["Asia"] == False, "EU")
            out_rows.append({"dow": int(dow), "dow_name": _dow_name(int(dow)),
                             "relation": "P(EU bull | Asia bull)", "prob_pct": p1, "N": n1})
            out_rows.append({"dow": int(dow), "dow_name": _dow_name(int(dow)),
                             "relation": "P(EU bull | Asia bear)", "prob_pct": p2, "N": n2})

        # US | EU
        if "EU" in part.columns and "US" in part.columns:
            p3, n3 = _prob(part["EU"] == True, "US")
            p4, n4 = _prob(part["EU"] == False, "US")
            out_rows.append({"dow": int(dow), "dow_name": _dow_name(int(dow)),
                             "relation": "P(US bull | EU bull)", "prob_pct": p3, "N": n3})
            out_rows.append({"dow": int(dow), "dow_name": _dow_name(int(dow)),
                             "relation": "P(US bull | EU bear)", "prob_pct": p4, "N": n4})

    return pd.DataFrame(out_rows)


def _day_to_day_conditionals(daily: pd.DataFrame) -> pd.DataFrame:
    """
    P(today bull | yesterday bull/bear) по каждому дню недели.
    """
    d = daily.sort_values("date").copy()
    d["prev_bull"] = d["day_bull"].shift(1)

    out_rows: List[Dict[str, Any]] = []
    for dow, part in d.groupby("dow", sort=True):
        part = part.dropna(subset=["prev_bull"])
        if part.empty:
            continue

        def _prob(prev_val: bool) -> Tuple[float, int]:
            sub = part.loc[part["prev_bull"] == prev_val, "day_bull"]
            if len(sub) == 0:
                return (np.nan, 0)
            return (float(sub.mean() * 100.0), int(len(sub)))

        p1, n1 = _prob(True)
        p2, n2 = _prob(False)
        out_rows.append({"dow": int(dow), "dow_name": _dow_name(int(dow)),
                         "relation": "P(today bull | yesterday bull)", "prob_pct": p1, "N": n1})
        out_rows.append({"dow": int(dow), "dow_name": _dow_name(int(dow)),
                         "relation": "P(today bull | yesterday bear)", "prob_pct": p2, "N": n2})

    return pd.DataFrame(out_rows)


# -----------------------------
# Плоты
# -----------------------------
def _plot_heatmap_mean_ret_by_dow_hour(df: pd.DataFrame, save_path: str, dpi: int) -> None:
    # среднее ret_h по (dow, hour)
    pivot = df.pivot_table(index="dow", columns="hour", values="ret_h", aggfunc="mean").sort_index()
    data = pivot.values * 100.0  # проценты

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(data, aspect="auto")
    ax.set_title("Mean hourly return by DOW × Hour (UTC), %")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Day of Week")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([_dow_name(int(x)) for x in pivot.index])
    ax.set_xticks(range(0, 24, 1))
    ax.set_xticklabels([str(h) for h in range(24)], rotation=0)
    fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_lines_mean_ret_by_hour_dow(df: pd.DataFrame, save_path: str, dpi: int) -> None:
    g = df.groupby(["dow", "hour"], sort=True)["ret_h"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    for dow in sorted(g["dow"].unique()):
        part = g[g["dow"] == dow]
        ax.plot(part["hour"], part["ret_h"] * 100.0, label=_dow_name(int(dow)))
    ax.set_title("Mean hourly return profile by DOW (UTC), %")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Mean return, %")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=7, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_bars_session_mean_by_dow(session_by_dow: pd.DataFrame, save_path: str, dpi: int) -> None:
    piv = session_by_dow.pivot_table(index="dow_name", columns="session", values="mean_session_ret_pct", aggfunc="first")
    piv = piv.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]).dropna(how="all")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(piv.index))
    width = 0.25
    cols = list(piv.columns)
    for i, c in enumerate(cols):
        ax.bar(x + (i - (len(cols)-1)/2) * width, piv[c].values, width=width, label=c)

    ax.set_title("Mean session return by DOW (UTC), %")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("Mean session return, %")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_daily_mean_and_range(daily_by_dow: pd.DataFrame, save_path: str, dpi: int) -> None:
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d = daily_by_dow.set_index("dow_name").reindex(order).dropna(how="all").reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(d))
    width = 0.35

    ax.bar(x - width/2, d["mean_day_ret_pct"].values, width=width, label="Mean day return, %")
    ax.bar(x + width/2, d["mean_day_range_pct"].values, width=width, label="Mean day range, %")

    ax.set_title("Daily mean return vs mean range by DOW (UTC)")
    ax.set_xticks(x)
    ax.set_xticklabels(d["dow_name"].values)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _plot_conditionals_overall(sess: pd.DataFrame, save_path: str, dpi: int) -> None:
    piv = sess.pivot_table(index=["date"], columns="session", values="session_bull", aggfunc="last").reset_index()

    rows = []
    if "Asia" in piv.columns and "EU" in piv.columns:
        sub1 = piv.loc[piv["Asia"] == True, "EU"].dropna()
        sub2 = piv.loc[piv["Asia"] == False, "EU"].dropna()
        rows.append(("P(EU bull | Asia bull)", float(sub1.mean() * 100.0) if len(sub1) else np.nan))
        rows.append(("P(EU bull | Asia bear)", float(sub2.mean() * 100.0) if len(sub2) else np.nan))

    if "EU" in piv.columns and "US" in piv.columns:
        sub3 = piv.loc[piv["EU"] == True, "US"].dropna()
        sub4 = piv.loc[piv["EU"] == False, "US"].dropna()
        rows.append(("P(US bull | EU bull)", float(sub3.mean() * 100.0) if len(sub3) else np.nan))
        rows.append(("P(US bull | EU bear)", float(sub4.mean() * 100.0) if len(sub4) else np.nan))

    if not rows:
        return

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_title("Conditional bull probabilities across sessions (overall), %")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Probability, %")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


# -----------------------------
# Главная функция отчёта
# -----------------------------
def generate_dow_sessions_report(
    *,
    csv_path: str,
    out_dir: str = OUT_DIR_DEFAULT,
    plot_dpi: int = PLOT_DPI_DEFAULT,
    include_weekends: bool = False,
) -> Dict[str, Any]:
    paths = _ensure_dirs(out_dir)

    df = load_ohlcv_1h(csv_path)

    if not include_weekends:
        df = df[df["dow"].isin([0, 1, 2, 3, 4])].copy()

    daily = _compute_daily(df)
    sess = _compute_sessions_per_day(df)

    daily_by_dow = _agg_daily_by_dow(daily)
    session_by_dow = _agg_sessions_by_dow(sess)
    session_conditionals = _session_conditionals_by_dow(sess)
    day_to_day_conditionals = _day_to_day_conditionals(daily)

    # Сохраняем таблицы
    daily_by_dow.to_csv(os.path.join(paths.out_dir, "daily_by_dow.csv"), index=False)
    session_by_dow.to_csv(os.path.join(paths.out_dir, "session_by_dow.csv"), index=False)
    session_conditionals.to_csv(os.path.join(paths.out_dir, "session_conditionals_by_dow.csv"), index=False)
    day_to_day_conditionals.to_csv(os.path.join(paths.out_dir, "day_to_day_conditionals_by_dow.csv"), index=False)

    # TOP hours по дням (средний ret_h)
    top_rows: List[Dict[str, Any]] = []
    g = df.groupby(["dow", "hour"], sort=True)["ret_h"].mean().reset_index()
    for dow in sorted(g["dow"].unique()):
        part = g[g["dow"] == dow].sort_values("ret_h", ascending=False)
        for rank, (_, r) in enumerate(part.head(5).iterrows(), start=1):
            top_rows.append({
                "dow": int(dow),
                "dow_name": _dow_name(int(dow)),
                "rank": rank,
                "hour_utc": int(r["hour"]),
                "mean_ret_pct": float(r["ret_h"] * 100.0),
            })
    top_hours = pd.DataFrame(top_rows)
    top_hours.to_csv(os.path.join(paths.out_dir, "top_hours_by_dow.csv"), index=False)

    # Плоты
    _plot_heatmap_mean_ret_by_dow_hour(df, os.path.join(paths.img_dir, "01_heatmap_mean_ret_by_dow_hour.png"), plot_dpi)
    _plot_lines_mean_ret_by_hour_dow(df, os.path.join(paths.img_dir, "02_lines_mean_ret_by_hour_dow.png"), plot_dpi)
    _plot_bars_session_mean_by_dow(session_by_dow, os.path.join(paths.img_dir, "03_bars_mean_session_ret_by_dow.png"), plot_dpi)
    _plot_daily_mean_and_range(daily_by_dow, os.path.join(paths.img_dir, "04_daily_mean_ret_and_range_by_dow.png"), plot_dpi)
    _plot_conditionals_overall(sess, os.path.join(paths.img_dir, "05_conditional_prob_sessions_overall.png"), plot_dpi)

    # Markdown отчёт (краткий, но структурный)
    md_path = os.path.join(paths.out_dir, "REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BTC DOW × Sessions Report (UTC)\n\n")
        f.write(f"Source CSV: `{csv_path}`\n\n")
        f.write("Sessions (UTC): Asia 00–07, EU 08–15, US 16–23\n\n")

        f.write("## Daily by DOW\n\n")
        f.write(daily_by_dow.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Sessions by DOW\n\n")
        f.write(session_by_dow.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Session conditionals by DOW\n\n")
        if not session_conditionals.empty:
            f.write(session_conditionals.to_markdown(index=False))
        else:
            f.write("_Not enough data for conditionals._")
        f.write("\n\n")

        f.write("## Day-to-day conditionals by DOW\n\n")
        if not day_to_day_conditionals.empty:
            f.write(day_to_day_conditionals.to_markdown(index=False))
        else:
            f.write("_Not enough data for day-to-day conditionals._")
        f.write("\n\n")

        f.write("## Top 5 hours by mean hourly return (per DOW)\n\n")
        f.write(top_hours.to_markdown(index=False))
        f.write("\n")

    return {
        "out_dir": paths.out_dir,
        "img_dir": paths.img_dir,
        "daily_by_dow": daily_by_dow,
        "session_by_dow": session_by_dow,
        "session_conditionals_by_dow": session_conditionals,
        "day_to_day_conditionals_by_dow": day_to_day_conditionals,
        "top_hours_by_dow": top_hours,
    }


# -----------------------------
# Пример запуска (без CLI)
# -----------------------------
def example_run() -> None:
    # Поставьте свой путь:
    CSV_PATH = "/home/jupiter/PYTHON/MARKET_DATA/_crypto_data/BTCUSDT/BTCUSDT_1h.csv"

    res = generate_dow_sessions_report(
        csv_path=CSV_PATH,
        out_dir=OUT_DIR_DEFAULT,
        plot_dpi=PLOT_DPI_DEFAULT,
        include_weekends=False,
    )
    print("OK. Report saved to:", res["out_dir"])
    print("Images saved to:", res["img_dir"])


if __name__ == "__main__":
    example_run()
