from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Result container
# -----------------------------
@dataclass(frozen=True)
class MetricsTimelineResult:
    df_plot: pd.DataFrame
    backend: str
    fig: Any
    axes: Optional[Sequence[Any]] = None
    crosshair: Optional[Any] = None


# -----------------------------
# Matplotlib crosshair helper
# -----------------------------
class _MultiAxisCrosshair:
    def __init__(
        self,
        *,
        fig: plt.Figure,
        axes: Sequence[plt.Axes],
        df_plot: pd.DataFrame,
        metric_cols: Sequence[str],
    ) -> None:
        self.fig = fig
        self.axes = list(axes)
        self.df_plot = df_plot
        self.metric_cols = list(metric_cols)

        idx = df_plot.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("df_plot index must be DatetimeIndex.")

        idx_utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
        self._x_dt = idx_utc.to_pydatetime()
        self._x_num = mdates.date2num(self._x_dt)

        # One vertical line per axis
        self._vlines = [ax.axvline(self._x_num[0], linewidth=1.0, alpha=0.35, linestyle="--") for ax in self.axes]

        # Markers (one per plotted series group)
        self._markers = self._init_markers()

        # Info panel on the first axis
        ax0 = self.axes[0]
        self._info = ax0.annotate(
            "",
            xy=(0.995, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.85),
        )
        self._info.set_visible(False)

        self._cid_move: Optional[int] = None
        self._cid_leave: Optional[int] = None

    def _init_markers(self) -> dict:
        """
        Markers follow the nearest timestamp:
          - price: close
          - iv: iv_call25 & iv_put25
          - rr: rr25
          - bf: bf25
          - availability: availability_all_4
        """
        ax_price, ax_iv, ax_rr, ax_bf, ax_av = self.axes

        markers: dict = {}

        # Helper: create marker that reuses existing line color if possible
        def mk_marker(ax: plt.Axes, ref_line: Optional[Any] = None):
            color = None
            if ref_line is not None:
                try:
                    color = ref_line.get_color()
                except Exception:
                    color = None
            (m,) = ax.plot([], [], marker="o", linestyle="None", markersize=5, color=color)
            return m

        # price marker uses first line on that axis (close)
        ref_price_line = ax_price.lines[0] if ax_price.lines else None
        markers["close"] = mk_marker(ax_price, ref_price_line)

        # iv markers use first two lines on that axis
        ref_iv1 = ax_iv.lines[0] if len(ax_iv.lines) >= 1 else None
        ref_iv2 = ax_iv.lines[1] if len(ax_iv.lines) >= 2 else None
        markers["iv_call25"] = mk_marker(ax_iv, ref_iv1)
        markers["iv_put25"] = mk_marker(ax_iv, ref_iv2)

        ref_rr = ax_rr.lines[0] if ax_rr.lines else None
        markers["rr25"] = mk_marker(ax_rr, ref_rr)

        ref_bf = ax_bf.lines[0] if ax_bf.lines else None
        markers["bf25"] = mk_marker(ax_bf, ref_bf)

        ref_av = ax_av.lines[0] if ax_av.lines else None
        markers["availability_all_4"] = mk_marker(ax_av, ref_av)

        for m in markers.values():
            m.set_visible(False)

        return markers

    def connect(self) -> None:
        if self._cid_move is None:
            self._cid_move = self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        if self._cid_leave is None:
            self._cid_leave = self.fig.canvas.mpl_connect("figure_leave_event", self._on_leave)

    def disconnect(self) -> None:
        if self._cid_move is not None:
            self.fig.canvas.mpl_disconnect(self._cid_move)
            self._cid_move = None
        if self._cid_leave is not None:
            self.fig.canvas.mpl_disconnect(self._cid_leave)
            self._cid_leave = None

    def _hide(self) -> None:
        for vl in self._vlines:
            vl.set_visible(False)
        for m in self._markers.values():
            m.set_visible(False)
        self._info.set_visible(False)
        self.fig.canvas.draw_idle()

    def _on_leave(self, event) -> None:
        self._hide()

    def _on_move(self, event) -> None:
        if event.inaxes is None:
            self._hide()
            return

        if event.xdata is None:
            self._hide()
            return

        # Only react if mouse is over one of our axes
        if event.inaxes not in self.axes:
            self._hide()
            return

        x = float(event.xdata)

        # nearest index by x
        pos = int(np.searchsorted(self._x_num, x))
        if pos <= 0:
            i = 0
        elif pos >= len(self._x_num):
            i = len(self._x_num) - 1
        else:
            left = self._x_num[pos - 1]
            right = self._x_num[pos]
            i = pos - 1 if abs(x - left) <= abs(x - right) else pos

        x_i = self._x_num[i]
        dt_i = self._x_dt[i]

        # Update vlines
        for vl in self._vlines:
            vl.set_xdata([x_i, x_i])
            vl.set_visible(True)

        # Pull row values
        row = self.df_plot.iloc[i]
        close = row.get("close", np.nan)

        ivc = row.get("iv_call25", np.nan)
        ivp = row.get("iv_put25", np.nan)
        rr = row.get("rr25", np.nan)
        bf = row.get("bf25", np.nan)
        av = row.get("availability_all_4", np.nan)

        # Update markers (hide if NaN)
        def set_marker(name: str, ax_idx: int, y: float) -> None:
            m = self._markers.get(name)
            if m is None:
                return
            if y is None or (isinstance(y, float) and np.isnan(y)):
                m.set_data([], [])
                m.set_visible(False)
                return
            m.set_data([x_i], [float(y)])
            m.set_visible(True)

        set_marker("close", 0, close)
        set_marker("iv_call25", 1, ivc)
        set_marker("iv_put25", 1, ivp)
        set_marker("rr25", 2, rr)
        set_marker("bf25", 3, bf)
        set_marker("availability_all_4", 4, av)

        # Info panel text (human-friendly; if NaN -> "NA")
        def fmt(v: Any, digits: int = 6) -> str:
            if v is None:
                return "NA"
            try:
                if isinstance(v, (float, np.floating)) and np.isnan(v):
                    return "NA"
                if isinstance(v, (int, np.integer)):
                    return str(int(v))
                if isinstance(v, (float, np.floating)):
                    return f"{float(v):.{digits}f}"
                return str(v)
            except Exception:
                return "NA"

        txt = (
            f"{dt_i.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"close: {fmt(close, 2)}\n"
            f"iv_call25: {fmt(ivc, 4)}   iv_put25: {fmt(ivp, 4)}\n"
            f"rr25: {fmt(rr, 6)}   bf25: {fmt(bf, 6)}\n"
            f"data_ok(all4): {fmt(av, 0)}"
        )
        self._info.set_text(txt)
        self._info.set_visible(True)

        self.fig.canvas.draw_idle()


# -----------------------------
# Data prep helpers
# -----------------------------
def _prepare_df_plot(
    candles_1h: Union[np.ndarray, Sequence[Sequence[float]]],
    metrics_csv_path: str,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, Sequence[str]]:
    c = np.asarray(candles_1h)
    if c.ndim != 2 or c.shape[1] < 6:
        raise ValueError("candles_1h must be 2D with columns: [timestamp_ms, open, high, low, close, volume].")

    df_c = pd.DataFrame(
        c[:, :6],
        columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
    )
    df_c["timestamp_ms"] = pd.to_numeric(df_c["timestamp_ms"], errors="coerce").astype("Int64")
    df_c["dt"] = pd.to_datetime(df_c["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    df_c = df_c.dropna(subset=["dt"]).sort_values("dt").drop_duplicates(subset=["dt"], keep="last")
    df_c = df_c.set_index("dt")

    df_m = pd.read_csv(metrics_csv_path)
    required = {"timestamp_ms", "iv_call25", "iv_put25", "rr25", "bf25"}
    missing_cols = required.difference(df_m.columns)
    if missing_cols:
        raise ValueError(f"metrics csv is missing columns: {sorted(missing_cols)}")

    df_m["timestamp_ms"] = pd.to_numeric(df_m["timestamp_ms"], errors="coerce")
    df_m["dt"] = pd.to_datetime(df_m["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    df_m = df_m.dropna(subset=["dt"]).sort_values("dt")

    metric_cols = ["iv_call25", "iv_put25", "rr25", "bf25"]
    for col in metric_cols:
        df_m[col] = pd.to_numeric(df_m[col], errors="coerce")

    # 0 считаем как “нет данных”
    df_m[metric_cols] = df_m[metric_cols].replace(0, np.nan)

    df_m = df_m.drop_duplicates(subset=["dt"], keep="last").set_index("dt")

    if df_m[metric_cols].dropna(how="all").empty:
        raise ValueError("No non-zero metrics found in CSV (everything is missing/zero).")

    start = df_m.index.min().floor("H")
    end = df_m.index.max().floor("H")
    idx = pd.date_range(start=start, end=end, freq="H", tz="UTC")

    df_close = df_c[["close"]].reindex(idx)
    df_metrics = df_m[metric_cols].reindex(idx)

    df_plot = pd.concat([df_close, df_metrics], axis=1)

    availability = (~df_plot[metric_cols].isna()).all(axis=1)
    df_plot["availability_all_4"] = availability.astype(int)

    return df_plot, start, end, metric_cols


def _compute_no_data_blocks(df_plot: pd.DataFrame, metric_cols: Sequence[str]) -> list[Tuple[pd.Timestamp, pd.Timestamp]]:
    availability = (~df_plot[list(metric_cols)].isna()).all(axis=1)
    no_data = ~availability
    no_data_blocks: list[Tuple[pd.Timestamp, pd.Timestamp]] = []

    if not no_data.any():
        return no_data_blocks

    s = no_data.astype(int)
    changes = s.diff().fillna(0)

    starts = df_plot.index[changes == 1].tolist()
    ends = df_plot.index[changes == -1].tolist()

    if no_data.iloc[0]:
        starts = [df_plot.index[0]] + starts
    if no_data.iloc[-1]:
        ends = ends + [df_plot.index[-1] + pd.Timedelta(hours=1)]

    for a, b in zip(starts, ends):
        no_data_blocks.append((a, b))

    return no_data_blocks


# -----------------------------
# Main plotting function
# -----------------------------
def plot_metrics_mini_timeline(
    candles_1h: Union[np.ndarray, Sequence[Sequence[float]]],
    metrics_csv_path: str = "metrics_mini.csv",
    *,
    backend: str = "plotly",  # "plotly" or "matplotlib"
    title: Optional[str] = None,
    show: bool = True,
    matplotlib_interactive_crosshair: bool = True,
) -> MetricsTimelineResult:
    """
    Интерактивная визуализация метрик iv_call25, iv_put25, rr25, bf25 на таймлайне.

    backend:
      - "plotly": максимально интерактивно (вертикальная линия по всем графикам, unified hover, зум/пан).
      - "matplotlib": остаёмся на matplotlib, но добавляем crosshair (вертикальная линия на всех осях + панель значений).

    Пропуски:
      - если метрик нет (NaN) или значение было 0 в csv → считаем “нет данных” и линия рвётся.
      - интервалы “нет данных” подсвечиваются полупрозрачными полосами.
    """
    df_plot, start, end, metric_cols = _prepare_df_plot(candles_1h, metrics_csv_path)
    no_data_blocks = _compute_no_data_blocks(df_plot, metric_cols)

    if title is None:
        title = f"IV/RR/BF timeline ({start.strftime('%Y-%m-%d %H:%M')} → {end.strftime('%Y-%m-%d %H:%M')} UTC)"

    backend_norm = (backend or "").strip().lower()
    if backend_norm not in {"plotly", "matplotlib"}:
        raise ValueError('backend must be "plotly" or "matplotlib".')

    if backend_norm == "plotly":
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except Exception as e:
            raise ImportError(
                "Plotly is not installed. Install it with: pip install plotly\n"
                "Or use backend='matplotlib'."
            ) from e

        fig = make_subplots(
            rows=5,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.28, 0.28, 0.16, 0.16, 0.12],
            subplot_titles=("Close", "IV 25", "RR 25", "BF 25", "Data OK"),
        )

        x = df_plot.index

        fig.add_trace(go.Scatter(x=x, y=df_plot["close"], mode="lines", name="close"), row=1, col=1)

        fig.add_trace(go.Scatter(x=x, y=df_plot["iv_call25"], mode="lines", name="iv_call25"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=df_plot["iv_put25"], mode="lines", name="iv_put25"), row=2, col=1)

        fig.add_trace(go.Scatter(x=x, y=df_plot["rr25"], mode="lines", name="rr25"), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=df_plot["bf25"], mode="lines", name="bf25"), row=4, col=1)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=df_plot["availability_all_4"],
                mode="lines",
                name="data_ok",
                line_shape="hv",
            ),
            row=5,
            col=1,
        )

        # shaded "no data" blocks
        for (a, b) in no_data_blocks:
            fig.add_vrect(
                x0=a,
                x1=b,
                fillcolor="rgba(120,120,120,0.15)",
                line_width=0,
                layer="below",
            )

        fig.update_layout(
            title=title,
            hovermode="x unified",  # unified tooltip for the whole vertical slice
            legend=dict(orientation="h"),
            margin=dict(l=60, r=30, t=60, b=40),
        )

        # vertical "spike" line across all subplots
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="solid",
        )

        # Make availability axis nice
        fig.update_yaxes(range=[-0.15, 1.15], row=5, col=1)

        if show:
            fig.show()

        return MetricsTimelineResult(df_plot=df_plot, backend="plotly", fig=fig, axes=None, crosshair=None)

    # -------------------------
    # Matplotlib backend (+ optional crosshair)
    # -------------------------
    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        sharex=True,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [2.2, 2.2, 1.2, 1.2, 0.8]},
    )

    ax_price, ax_iv, ax_rr, ax_bf, ax_av = axes

    ax_price.plot(df_plot.index, df_plot["close"], linewidth=1.2)
    ax_price.set_ylabel("Close")
    ax_price.grid(True, alpha=0.3)

    ax_iv.plot(df_plot.index, df_plot["iv_call25"], linewidth=1.2, label="iv_call25")
    ax_iv.plot(df_plot.index, df_plot["iv_put25"], linewidth=1.2, label="iv_put25")
    ax_iv.set_ylabel("IV 25")
    ax_iv.grid(True, alpha=0.3)
    ax_iv.legend(loc="upper left")

    ax_rr.plot(df_plot.index, df_plot["rr25"], linewidth=1.2)
    ax_rr.set_ylabel("RR 25")
    ax_rr.grid(True, alpha=0.3)

    ax_bf.plot(df_plot.index, df_plot["bf25"], linewidth=1.2)
    ax_bf.set_ylabel("BF 25")
    ax_bf.grid(True, alpha=0.3)

    ax_av.step(df_plot.index, df_plot["availability_all_4"], where="post", linewidth=1.2)
    ax_av.set_ylabel("Data\nOK")
    ax_av.set_yticks([0, 1])
    ax_av.set_ylim(-0.15, 1.15)
    ax_av.grid(True, axis="y", alpha=0.3)

    for (a, b) in no_data_blocks:
        for ax in axes:
            ax.axvspan(a, b, alpha=0.15)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_av.xaxis.set_major_locator(locator)
    ax_av.xaxis.set_major_formatter(formatter)

    fig.suptitle(title)
    fig.tight_layout()

    crosshair = None
    if matplotlib_interactive_crosshair:
        crosshair = _MultiAxisCrosshair(fig=fig, axes=axes, df_plot=df_plot, metric_cols=metric_cols)
        crosshair.connect()

    if show:
        plt.show()

    return MetricsTimelineResult(df_plot=df_plot, backend="matplotlib", fig=fig, axes=axes, crosshair=crosshair)


# -----------------------------
# Example (your style)
# -----------------------------
from datetime import datetime
import shared_vars as sv
from helpers.get_data import load_data_sets

sv.START = datetime(2020, 8, 1)
sv.END = datetime(2027, 1, 1)
sv.data_1h = load_data_sets(60)

res = plot_metrics_mini_timeline(
    candles_1h=sv.data_1h,
    metrics_csv_path="metrics_mini.csv",
    backend="plotly",
    show=True,
)
