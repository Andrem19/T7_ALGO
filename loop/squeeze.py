# -*- coding: utf-8 -*-
"""
Переписанная функция:
  1) Принимает свечи в формате numpy.ndarray (N, 6): [timestamp_ms, open, high, low, close, volume]
  2) Опционально сохраняет картинку (флаг save_image)
  3) Добавляет день недели к датам на оси X и в сводке
  4) ВОЗВРАЩАЕТ squeeze index (последнего бара): BB_width / KC_width

Зависимости: numpy, pandas, matplotlib, mplfinance, talib
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import talib as ta


def plot_kc_bb_squeeze_np(
    candles_np: np.ndarray,
    out_path: str,
    bb_period: int = 21,
    bb_dev: float = 2.0,
    kc_period: int = 21,
    kc_mult: float = 1.5,
    title: str | None = None,
    save_image: bool = True,
) -> float:
    """
    Строит график Bollinger vs Keltner (60×1h), считает индекс «сжатия»
    и (опционально) сохраняет изображение.

    Параметры:
        candles_np : np.ndarray формы (N, 6) [ts_ms, open, high, low, close, volume]
        out_path   : путь для сохранения изображения (используется, если save_image=True)
        bb_period  : период Bollinger Bands (по Close)
        bb_dev     : кол-во σ для BB
        kc_period  : период EMA/ATR для Keltner
        kc_mult    : множитель ATR для Keltner
        title      : заголовок графика
        save_image : сохранять ли картинку (True/False)

    Возвращает:
        float — значение squeeze index на последнем баре (BB_width / KC_width).
    """
    # ---- Валидация входа ----
    if not isinstance(candles_np, np.ndarray):
        raise TypeError("candles_np должен быть numpy.ndarray формы (N, 6).")
    if candles_np.ndim != 2 or candles_np.shape[1] != 6:
        raise ValueError("Ожидается массив формы (N, 6): [ts_ms, open, high, low, close, volume].")
    if candles_np.shape[0] < 60:
        raise ValueError("Нужно минимум 60 свечей (N >= 60).")

    # Берём последние 60 часов
    data = candles_np[-60:, :].astype(float, copy=True)

    # ---- Подготовка данных ----
    ts_ms = data[:, 0].astype(np.int64)
    o = data[:, 1]
    h = data[:, 2]
    l = data[:, 3]
    c = data[:, 4]

    # Индекс времени (UTC → naive)
    dt_index = pd.to_datetime(ts_ms, unit="ms", utc=True).tz_convert(None)

    df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=dt_index)

    # ---- Индикаторы (TA-Lib) ----
    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values

    bb_up, bb_mid, bb_dn = ta.BBANDS(
        close_arr, timeperiod=bb_period, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0
    )
    ema_mid = ta.EMA(close_arr, timeperiod=kc_period)
    atr     = ta.ATR(high_arr, low_arr, close_arr, timeperiod=kc_period)
    kc_up   = ema_mid + kc_mult * atr
    kc_dn   = ema_mid - kc_mult * atr

    # В Series с индексом дат
    bb_up  = pd.Series(bb_up,  index=df.index, name="BB_Upper")
    bb_mid = pd.Series(bb_mid, index=df.index, name="BB_Middle")
    bb_dn  = pd.Series(bb_dn,  index=df.index, name="BB_Lower")
    ema_s  = pd.Series(ema_mid,index=df.index, name="KC_Middle")
    atr_s  = pd.Series(atr,    index=df.index, name="ATR")
    kc_up  = pd.Series(kc_up,  index=df.index, name="KC_Upper")
    kc_dn  = pd.Series(kc_dn,  index=df.index, name="KC_Lower")

    # ---- Индекс «сжатия» ----
    bb_width = (bb_up - bb_dn)
    kc_width = (kc_up - kc_dn)
    price    = df["Close"]
    eps = 1e-12
    squeeze_ratio = (bb_width / (kc_width + eps))

    # Последнее значение индекса (NaN → nan float)
    last_ratio = float(squeeze_ratio.iloc[-1]) if not math.isnan(squeeze_ratio.iloc[-1]) else float("nan")
    squeeze_count = _count_consecutive_trues_from_end(((bb_up<=kc_up)&(bb_dn>=kc_dn)).values)
    # ---- Визуализация (опционально) ----
    if save_image:
        if title is None:
            title = "Squeeze: Bollinger vs Keltner (60×1h)"

        apds = [
            mpf.make_addplot(bb_up,  color="tab:blue",   width=1.0),
            mpf.make_addplot(bb_mid, color="tab:blue",   width=1.0, linestyle="--"),
            mpf.make_addplot(bb_dn,  color="tab:blue",   width=1.0),

            mpf.make_addplot(kc_up,  color="tab:orange", width=1.0),
            mpf.make_addplot(ema_s,  color="tab:orange", width=1.0, linestyle="--"),
            mpf.make_addplot(kc_dn,  color="tab:orange", width=1.0),
        ]

        # Добавляем день недели в подписи оси X через формат %a
        # (например, 2025-10-10 Fri 13:00)
        fig, axes = mpf.plot(
            df,
            type="candle",
            style="yahoo",
            addplot=apds,
            title=title,
            volume=False,
            returnfig=True,
            figscale=1.2,
            figratio=(16, 9),
            datetime_format="%Y-%m-%d (%a) %H:%M"  # <-- день недели
        )
        ax = axes[0]

        # Сводка в шапке (также с днём недели)
        last_dt = df.index[-1]
        # Ширины в % для наглядности
        bb_w_pct = float(((bb_up - bb_dn) / price * 100.0).iloc[-1])
        kc_w_pct = float(((kc_up - kc_dn) / price * 100.0).iloc[-1])
        
        

        summary_lines = [
            f"Последний бар: {last_dt.strftime('%Y-%m-%d (%a) %H:%M')}",
            f"BB width: {bb_w_pct:.2f}%   |   KC width: {kc_w_pct:.2f}%",
            f"Squeeze index (BB/KC): {last_ratio:.2f}",
            f"Squeeze ON: {'ДА' if (bb_up.iloc[-1] <= kc_up.iloc[-1]) and (bb_dn.iloc[-1] >= kc_dn.iloc[-1]) else 'НЕТ'}",
            f"Текущая серия сжатия: { squeeze_count } бар(ов)"
        ]
        ax.text(
            0.02, 0.98, "\n".join(summary_lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="lightgrey")
        )

        legend_text = (
            "Синие: Bollinger (верх/средняя/низ)\n"
            "Оранжевые: Keltner (верх/средняя/низ)\n"
            "Сжатие: BB внутри KC  →  ratio < 1"
        )
        ax.text(
            0.98, 0.02, legend_text,
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="lightgrey")
        )

        # Сохранение
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Возврат индекса «сжатия» последнего бара ----
    return last_ratio, squeeze_count


# -------------------------
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ
# -------------------------
def _count_consecutive_trues_from_end(arr: np.ndarray) -> int:
    """Считает, сколько True подряд с конца массива arr."""
    cnt = 0
    for v in arr[::-1]:
        if bool(v):
            cnt += 1
        else:
            break
    return cnt
