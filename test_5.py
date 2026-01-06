from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# Result containers
# ============================================================
@dataclass(frozen=True)
class MetricsTimelineResult:
    df_plot: pd.DataFrame
    backend: str
    fig: Any
    axes: Optional[Sequence[Any]] = None
    crosshair: Optional[Any] = None


@dataclass(frozen=True)
class SmileDashboardResult:
    df_smile: pd.DataFrame
    fig: plt.Figure
    axes: Sequence[plt.Axes]


# ============================================================
# Matplotlib crosshair helper (for 1st window, unchanged)
# ============================================================
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

        self._vlines = [ax.axvline(self._x_num[0], linewidth=1.0, alpha=0.35, linestyle="--") for ax in self.axes]
        self._markers = self._init_markers()

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
        ax_price, ax_iv, ax_rr, ax_bf, ax_av = self.axes
        markers: dict = {}

        def mk_marker(ax: plt.Axes, ref_line: Optional[Any] = None):
            color = None
            if ref_line is not None:
                try:
                    color = ref_line.get_color()
                except Exception:
                    color = None
            (m,) = ax.plot([], [], marker="o", linestyle="None", markersize=5, color=color)
            return m

        ref_price_line = ax_price.lines[0] if ax_price.lines else None
        markers["close"] = mk_marker(ax_price, ref_price_line)

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
        if event.inaxes is None or event.xdata is None:
            self._hide()
            return
        if event.inaxes not in self.axes:
            self._hide()
            return

        x = float(event.xdata)

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

        for vl in self._vlines:
            vl.set_xdata([x_i, x_i])
            vl.set_visible(True)

        row = self.df_plot.iloc[i]
        close = row.get("close", np.nan)
        ivc = row.get("iv_call25", np.nan)
        ivp = row.get("iv_put25", np.nan)
        rr = row.get("rr25", np.nan)
        bf = row.get("bf25", np.nan)
        av = row.get("availability_all_4", np.nan)

        def set_marker(name: str, y: float) -> None:
            m = self._markers.get(name)
            if m is None:
                return
            if y is None or (isinstance(y, float) and np.isnan(y)):
                m.set_data([], [])
                m.set_visible(False)
                return
            m.set_data([x_i], [float(y)])
            m.set_visible(True)

        set_marker("close", close)
        set_marker("iv_call25", ivc)
        set_marker("iv_put25", ivp)
        set_marker("rr25", rr)
        set_marker("bf25", bf)
        set_marker("availability_all_4", av)

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


# ============================================================
# Data prep helpers
# ============================================================
def _prepare_df_plot(
    candles_1h: Union[np.ndarray, Sequence[Sequence[float]]],
    metrics_csv_path: showing,
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


# ============================================================
# 1) First window: existing timeline function (unchanged behaviour)
# ============================================================
def plot_metrics_mini_timeline(
    candles_1h: Union[np.ndarray, Sequence[Sequence[float]]],
    metrics_csv_path: str = "metrics_mini.csv",
    *,
    backend: str = "plotly",  # "plotly" or "matplotlib"
    title: Optional[str] = None,
    show: bool = True,
    matplotlib_interactive_crosshair: bool = True,
) -> MetricsTimelineResult:
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
            hovermode="x unified",
            legend=dict(orientation="h"),
            margin=dict(l=60, r=30, t=60, b=40),
        )

        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="solid",
        )

        fig.update_yaxes(range=[-0.15, 1.15], row=5, col=1)

        if show:
            fig.show()

        return MetricsTimelineResult(df_plot=df_plot, backend="plotly", fig=fig, axes=None, crosshair=None)

    # matplotlib
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


# ============================================================
# 2) Second window: smile dashboard (separate)
# ============================================================
def _compute_smile_features(df_plot: pd.DataFrame) -> pd.DataFrame:
    df = df_plot.copy()

    # level proxy at 25d: average of call25 and put25
    df["iv_mid25"] = (df["iv_call25"] + df["iv_put25"]) / 2.0

    # approximate ATM IV using BF definition: bf25 = mid25 - atm  -> atm = mid25 - bf25
    df["iv_atm_est"] = df["iv_mid25"] - df["bf25"]

    # recompute rr from iv legs (quality check / alternative)
    df["rr_from_iv"] = df["iv_call25"] - df["iv_put25"]

    # how much rr25 differs from recomputed (should be near 0 if convention matches)
    df["rr_gap"] = df["rr25"] - df["rr_from_iv"]

    # normalised shape indicators
    denom = df["iv_atm_est"].replace(0, np.nan)
    df["rr25_norm"] = df["rr25"] / denom
    df["bf25_norm"] = df["bf25"] / denom

    # 3-point smile series
    df["smile_put25"] = df["iv_put25"]
    df["smile_atm"] = df["iv_atm_est"]
    df["smile_call25"] = df["iv_call25"]

    return df


def plot_smile_dashboard_matplotlib(
    df_plot: pd.DataFrame,
    *,
    title: str = "Smile dashboard (put25 / atm_est / call25)",
    show: bool = True,
) -> SmileDashboardResult:
    """
    Отдельное окно:
      - Smile curve (3 точки: put25, atm_est, call25) с ползунком времени
      - Heatmap этих 3 рядов по времени
      - Доп. ряды (iv_mid25, iv_atm_est, rr_gap, rr_norm, bf_norm)
    """
    from matplotlib.widgets import Slider

    if not isinstance(df_plot.index, pd.DatetimeIndex):
        raise ValueError("df_plot must have DatetimeIndex index.")

    df = _compute_smile_features(df_plot)

    # Для улыбки нам критично иметь call/put/bf (иначе atm_est будет NaN)
    smile_cols = ["smile_put25", "smile_atm", "smile_call25"]
    avail_smile = (~df[smile_cols].isna()).all(axis=1)

    # Prepare X as numeric for some elements
    idx = df.index
    idx_utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    x_dt = idx_utc.to_pydatetime()
    x_num = mdates.date2num(x_dt)

    n = len(df)
    if n < 2:
        raise ValueError("Not enough points to build dashboard.")

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.1, 1.0],
        width_ratios=[1.0, 1.25],
        left=0.06,
        right=0.98,
        top=0.92,
        bottom=0.12,
        wspace=0.18,
        hspace=0.22,
    )

    ax_smile = fig.add_subplot(gs[0, 0])
    ax_derived = fig.add_subplot(gs[1, 0], sharex=None)
    ax_heat = fig.add_subplot(gs[:, 1])
    axes = (ax_smile, ax_derived, ax_heat)

    fig.suptitle(title)

    # -------------------------
    # Smile plot (3 points)
    # -------------------------
    x_smile = np.array([-25.0, 0.0, 25.0], dtype=float)

    (line_smile,) = ax_smile.plot([], [], linewidth=1.6)
    (pts_smile,) = ax_smile.plot([], [], marker="o", linestyle="None", markersize=6)

    txt_no_data = ax_smile.text(
        0.5,
        0.5,
        "No data for this hour",
        transform=ax_smile.transAxes,
        ha="center",
        va="center",
        alpha=0.75,
    )
    txt_no_data.set_visible(False)

    ax_smile.set_title("Smile curve (25d put → ATM_est → 25d call)")
    ax_smile.set_xlabel("Delta (proxy)")
    ax_smile.set_ylabel("IV")
    ax_smile.set_xlim(-30, 30)
    ax_smile.grid(True, alpha=0.25)

    # -------------------------
    # Heatmap (3 x N)
    # -------------------------
    heat_data = np.vstack([df["smile_put25"].to_numpy(), df["smile_atm"].to_numpy(), df["smile_call25"].to_numpy()])
    heat_masked = np.ma.masked_invalid(heat_data)

    # Use date extent so x-axis can display real dates
    x0 = x_num[0]
    x1 = x_num[-1]
    im = ax_heat.imshow(
        heat_masked,
        aspect="auto",
        origin="lower",
        extent=[x0, x1, -0.5, 2.5],
        interpolation="nearest",
    )
    ax_heat.set_title("Smile levels heatmap (put25 / atm_est / call25)")
    ax_heat.set_yticks([0, 1, 2])
    ax_heat.set_yticklabels(["put25", "atm_est", "call25"])

    locator = mdates.AutoDateLocator(minticks=6, maxticks=14)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_heat.xaxis.set_major_locator(locator)
    ax_heat.xaxis.set_major_formatter(formatter)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cbar.set_label("IV")

    vline_heat = ax_heat.axvline(x_num[0], linewidth=1.0, alpha=0.5, linestyle="--")

    # -------------------------
    # Derived series plot
    # -------------------------
    ax_derived.set_title("Derived series (level / checks / normalised shape)")
    ax_derived.grid(True, alpha=0.25)
    ax_derived.set_xlabel("Time (UTC)")

    # Left axis: iv_mid25 and iv_atm_est
    (l_mid,) = ax_derived.plot(x_num, df["iv_mid25"].to_numpy(), linewidth=1.2, label="iv_mid25")
    (l_atm,) = ax_derived.plot(x_num, df["iv_atm_est"].to_numpy(), linewidth=1.2, label="iv_atm_est")
    ax_derived.set_ylabel("IV level")

    # Right axis: rr_gap + norms (optional but useful)
    ax_r = ax_derived.twinx()
    (l_gap,) = ax_r.plot(x_num, df["rr_gap"].to_numpy(), linewidth=1.0, alpha=0.9, label="rr_gap")
    (l_rrn,) = ax_r.plot(x_num, df["rr25_norm"].to_numpy(), linewidth=1.0, alpha=0.85, label="rr25_norm")
    (l_bfn,) = ax_r.plot(x_num, df["bf25_norm"].to_numpy(), linewidth=1.0, alpha=0.85, label="bf25_norm")
    ax_r.set_ylabel("RR/BF shape (norm / gap)")

    vline_der = ax_derived.axvline(x_num[0], linewidth=1.0, alpha=0.5, linestyle="--")
    vline_der_r = ax_r.axvline(x_num[0], linewidth=1.0, alpha=0.5, linestyle="--")

    # Legends: merge left+right
    h1, lab1 = ax_derived.get_legend_handles_labels()
    h2, lab2 = ax_r.get_legend_handles_labels()
    ax_derived.legend(h1 + h2, lab1 + lab2, loc="upper left")

    # Format x-axis on derived plot
    ax_derived.xaxis.set_major_locator(locator)
    ax_derived.xaxis.set_major_formatter(formatter)

    # -------------------------
    # Info box (top-left on smile)
    # -------------------------
    info = ax_smile.annotate(
        "",
        xy=(0.01, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.85),
    )

    def _fmt(v: Any, digits: int = 6) -> str:
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

    def update(i: int) -> None:
        i = int(max(0, min(n - 1, i)))
        row = df.iloc[i]
        dt_i = x_dt[i]
        x_i = x_num[i]

        vline_heat.set_xdata([x_i, x_i])
        vline_der.set_xdata([x_i, x_i])
        vline_der_r.set_xdata([x_i, x_i])

        if bool(avail_smile.iloc[i]):
            y = np.array([row["smile_put25"], row["smile_atm"], row["smile_call25"]], dtype=float)
            line_smile.set_data(x_smile, y)
            pts_smile.set_data(x_smile, y)
            line_smile.set_visible(True)
            pts_smile.set_visible(True)
            txt_no_data.set_visible(False)

            # reasonable y-limits around this smile
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            pad = max(1e-6, (y_max - y_min) * 0.25)
            ax_smile.set_ylim(y_min - pad, y_max + pad)
        else:
            line_smile.set_data([], [])
            pts_smile.set_data([], [])
            line_smile.set_visible(False)
            pts_smile.set_visible(False)
            txt_no_data.set_visible(True)

        txt = (
            f"{dt_i.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"close: {_fmt(row.get('close'), 2)}\n"
            f"iv_put25: {_fmt(row.get('iv_put25'), 4)}   iv_call25: {_fmt(row.get('iv_call25'), 4)}\n"
            f"iv_mid25: {_fmt(row.get('iv_mid25'), 4)}   iv_atm_est: {_fmt(row.get('iv_atm_est'), 4)}\n"
            f"rr25: {_fmt(row.get('rr25'), 6)}   rr_from_iv: {_fmt(row.get('rr_from_iv'), 6)}   rr_gap: {_fmt(row.get('rr_gap'), 6)}\n"
            f"bf25: {_fmt(row.get('bf25'), 6)}   rr_norm: {_fmt(row.get('rr25_norm'), 6)}   bf_norm: {_fmt(row.get('bf25_norm'), 6)}\n"
            f"data_ok(all4): {_fmt(row.get('availability_all_4'), 0)}"
        )
        info.set_text(txt)

        fig.canvas.draw_idle()

    # -------------------------
    # Slider
    # -------------------------
    ax_slider = fig.add_axes([0.10, 0.05, 0.75, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="t (index)",
        valmin=0,
        valmax=n - 1,
        valinit=0,
        valstep=1,
    )

    def _on_slider(val: float) -> None:
        update(int(val))

    slider.on_changed(_on_slider)

    # init
    update(0)

    if show:
        plt.show()

    return SmileDashboardResult(df_smile=df, fig=fig, axes=axes)


# ============================================================
# Convenience wrapper: show first window, then second
# ============================================================
def show_metrics_then_smile_dashboard(
    candles_1h: Union[np.ndarray, Sequence[Sequence[float]]],
    metrics_csv_path: str = "metrics_mini.csv",
    *,
    backend_timeline: str = "plotly",
    show: bool = True,
    plotly_wait: bool = False,
) -> Tuple[MetricsTimelineResult, SmileDashboardResult]:
    """
    1) Показывает первое окно (таймлайн) как раньше.
    2) Затем (после закрытия первого — если matplotlib; либо после ожидания — если plotly_wait=True)
       показывает второе окно со smile dashboard.

    plotly_wait:
      - если backend_timeline="plotly", то fig.show() обычно не блокирует выполнение.
        Чтобы второе окно не появлялось сразу, можно включить plotly_wait=True (будет ожидание Enter).
    """
    res1 = plot_metrics_mini_timeline(
        candles_1h=candles_1h,
        metrics_csv_path=metrics_csv_path,
        backend=backend_timeline,
        show=show,
    )

    if show and (backend_timeline or "").strip().lower() == "plotly" and plotly_wait:
        try:
            input("Close the first Plotly tab/window, then press Enter to open Smile dashboard...")
        except Exception:
            pass

    res2 = plot_smile_dashboard_matplotlib(
        res1.df_plot,
        title="Smile dashboard (put25 / atm_est / call25) + derived checks",
        show=show,
    )
    return res1, res2


# ============================================================
# Example (your style)
# ============================================================
if __name__ == "__main__":
    from datetime import datetime
    import shared_vars as sv
    from helpers.get_data import load_data_sets

    sv.START = datetime(2020, 8, 1)
    sv.END = datetime(2027, 1, 1)
    sv.data_1h = load_data_sets(60)

    # Вариант 1: plotly таймлайн + потом smile (для plotly лучше включить plotly_wait=True)
    show_metrics_then_smile_dashboard(
        candles_1h=sv.data_1h,
        metrics_csv_path="metrics_mini.csv",
        backend_timeline="plotly",
        show=True,
        plotly_wait=True,
    )

    # Вариант 2: matplotlib таймлайн (закрыли окно → открылось второе автоматически)
    # show_metrics_then_smile_dashboard(
    #     candles_1h=sv.data_1h,
    #     metrics_csv_path="metrics_mini.csv",
    #     backend_timeline="matplotlib",
    #     show=True,
    # )
