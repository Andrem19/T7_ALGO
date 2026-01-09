from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib as mpl
import time as _time

# ВАЖНО:
# Чтобы не цеплялся tkinter (TkAgg) и не было ошибок "main thread is not in main loop",
# используем headless backend для сохранения PNG и HTML-viewer.
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.colors import TwoSlopeNorm
# =============================================================================
# CONFIG
# =============================================================================

REPORT_DIR_DEFAULT = "_trade_report"
REPORT_SERVER_HOST = "127.0.0.1"
REPORT_SERVER_PORT = 8765
REPORT_SERVER_PORT_TRIES = 20
_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ENV_PATH_DEFAULT = ".env"


@dataclass
class DrawdownInfo:
    mdd: float                  # величина максимальной просадки (>0)
    peak_idx: int               # индекс пика (до начала просадки)
    trough_idx: int             # индекс дна (минимум в просадке)
    start_idx: int              # индекс начала просадки (совпадает с peak_idx)
    end_idx: int                # индекс конца просадки (совпадает с trough_idx)
# =============================================================================
# .env loader (без зависимостей)
# =============================================================================

def _load_dotenv_simple(dotenv_path: str = ENV_PATH_DEFAULT) -> None:
    """
    Минимальная загрузка .env:
    - читает KEY=VALUE
    - игнорирует пустые строки и комментарии
    - если переменная уже есть в окружении, не перезаписывает
    """
    import os

    if not dotenv_path:
        return
    if not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k:
                    continue
                if os.getenv(k) is None:
                    os.environ[k] = v
    except Exception:
        # намеренно тихо: отчёт должен строиться даже без телеги
        return


# =============================================================================
# Telegram sender (через stdlib urllib, без requests)
# =============================================================================

def _telegram_send_photo(
    *,
    token: str,
    chat_id: str,
    file_path: str,
    caption: Optional[str] = None,
    timeout_sec: int = 30,
) -> None:
    """
    Отправляет PNG/JPG как photo в Telegram.
    Бросает исключение при ошибке.
    """
    import os
    import json
    import uuid
    import mimetypes
    import urllib.request

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    boundary = "----tgform" + uuid.uuid4().hex
    crlf = "\r\n"

    def part_text(name: str, value: str) -> bytes:
        return (
            f"--{boundary}{crlf}"
            f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'
            f"{value}{crlf}"
        ).encode("utf-8")

    def part_file(name: str, path: str) -> bytes:
        filename = os.path.basename(path)
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        head = (
            f"--{boundary}{crlf}"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"{crlf}'
            f"Content-Type: {ctype}{crlf}{crlf}"
        ).encode("utf-8")
        tail = crlf.encode("utf-8")
        return head + data + tail

    body = b""
    body += part_text("chat_id", str(chat_id))
    if caption:
        body += part_text("caption", caption)
    body += part_file("photo", file_path)
    body += f"--{boundary}--{crlf}".encode("utf-8")

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = resp.read().decode("utf-8", errors="replace")

    try:
        j = json.loads(payload)
    except Exception:
        raise RuntimeError(f"Telegram API: non-JSON response: {payload[:400]}")

    if not j.get("ok", False):
        raise RuntimeError(f"Telegram API error: {payload}")


# =============================================================================
# Report JS manifest
# =============================================================================

def _write_images_js(report_dir: str, image_files: List[str]) -> str:
    """
    Пишет report_dir/images.js:
      window.REPORT_IMAGES = ["01_equity.png", ...]
    Возвращает путь к файлу.
    """
    import os
    import json

    names = [os.path.basename(x) for x in image_files]
    out = "window.REPORT_IMAGES = " + json.dumps(names, ensure_ascii=False) + ";\n"

    js_path = os.path.join(report_dir, "images.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(out)
    return js_path


# =============================================================================
# Save helpers (как раньше, через tmp)
# =============================================================================

def _run_plot_save_to_tmp_and_move(
    *,
    plot_fn: Callable[[str], None],
    tmp_dir: str,
    out_dir: str,
    base_name: str,
) -> List[str]:
    """
    1) чистит tmp_dir
    2) plot_fn(tmp_dir) обязан сохранить PNG в tmp_dir
    3) переносит PNG в out_dir с фиксированными именами
    """
    import os
    import glob
    import shutil

    os.makedirs(tmp_dir, exist_ok=True)

    # чистим tmp
    for fp in glob.glob(os.path.join(tmp_dir, "*.png")):
        try:
            os.remove(fp)
        except Exception:
            pass

    plot_fn(tmp_dir)

    pngs = sorted(glob.glob(os.path.join(tmp_dir, "*.png")))
    out_files: List[str] = []

    if not pngs:
        return out_files

    os.makedirs(out_dir, exist_ok=True)

    if len(pngs) == 1:
        dst = os.path.join(out_dir, f"{base_name}.png")
        try:
            os.remove(dst)
        except Exception:
            pass
        shutil.move(pngs[0], dst)
        out_files.append(dst)
        return out_files

    for i, src in enumerate(pngs, start=1):
        dst = os.path.join(out_dir, f"{base_name}_{i:02d}.png")
        try:
            os.remove(dst)
        except Exception:
            pass
        shutil.move(src, dst)
        out_files.append(dst)

    return out_files


# =============================================================================
# Local server (serves static + /api/send)
# =============================================================================

def _start_report_server(
    *,
    report_dir: str,
    host: str = REPORT_SERVER_HOST,
    port: int = REPORT_SERVER_PORT,
    port_tries: int = REPORT_SERVER_PORT_TRIES,
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
):
    """
    Запускает локальный HTTP сервер (static + /api/send + /api/shutdown) в отдельном daemon-thread.

    Возвращает:
      (base_url, stop_event)

    stop_event выставляется, когда пользователь нажал /api/shutdown
    (или если сервер остановили извне).
    """
    import os
    import json
    import threading
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    report_dir = os.path.abspath(report_dir)
    stop_event = threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            # CORS (как было)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

            # КРИТИЧНО: отключаем кэш браузера, иначе images.js / index.html легко залипают
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")

            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.end_headers()

        def _json(self, code: int, obj: Dict[str, Any]) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            u = urlparse(self.path)

            # --- shutdown viewer ---
            if u.path == "/api/shutdown":
                self._json(200, {"ok": True})
                stop_event.set()

                # shutdown вызываем из отдельного потока, чтобы не повесить handler
                def _shutdown():
                    try:
                        self.server.shutdown()
                    except Exception:
                        pass

                threading.Thread(target=_shutdown, daemon=True).start()
                return

            # --- send to telegram ---
            if u.path == "/api/send":
                if not telegram_token or not telegram_chat_id:
                    return self._json(400, {"ok": False, "error": "ALGO_BOT or CHAT_ID is not configured"})

                qs = parse_qs(u.query or "")
                fname = (qs.get("file", [""])[0] or "").strip()

                if not fname.lower().endswith(".png"):
                    return self._json(400, {"ok": False, "error": "Only .png is allowed"})

                safe = os.path.basename(fname)
                fpath = os.path.join(report_dir, safe)

                if not os.path.exists(fpath):
                    return self._json(404, {"ok": False, "error": f"File not found: {safe}"})

                try:
                    _telegram_send_photo(
                        token=telegram_token,
                        chat_id=telegram_chat_id,
                        file_path=fpath,
                        caption=safe,
                    )
                    return self._json(200, {"ok": True, "sent": safe})
                except Exception as e:
                    return self._json(500, {"ok": False, "error": str(e)})

            return super().do_POST()

        # чтобы меньше мусора в консоли (по желанию)
        def log_message(self, format, *args):
            return

    # чтобы порт можно было быстро переиспользовать после падения
    class ReuseThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    last_err: Optional[Exception] = None

    for p in range(int(port), int(port) + max(1, int(port_tries))):
        try:
            httpd = ReuseThreadingHTTPServer(
                (host, p),
                lambda *args, **kwargs: Handler(*args, directory=report_dir, **kwargs),
            )

            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()

            base_url = f"http://{host}:{p}"
            return base_url, stop_event

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not start report server on {host}:{port}.. ({last_err})")


def _is_interactive_session() -> bool:
    import sys
    if getattr(sys, "ps1", None) is not None:
        return True
    if getattr(sys.flags, "interactive", 0):
        return True
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

def _maybe_save_or_show(
    fig: plt.Figure,
    fname: str,
    show: bool,
    save: bool,
    save_dir: Optional[str],
    dpi: int,
) -> None:
    if save:
        if not save_dir:
            raise ValueError("Указан save=True, но не задан 'save_dir'.")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _annotate_mdd(
    ax: plt.Axes,
    x: np.ndarray,
    eq: np.ndarray,
    dd: DrawdownInfo,
    label_prefix: str,
) -> None:
    """Выделить зону максимальной просадки на графике."""
    if dd.mdd <= 0 or dd.end_idx <= dd.start_idx or dd.end_idx >= len(x):
        return
    ax.fill_between(
        x[dd.start_idx: dd.end_idx + 1],
        eq[dd.start_idx: dd.end_idx + 1],
        np.maximum.accumulate(eq[dd.start_idx: dd.end_idx + 1]),
        alpha=0.20,
        label=f"{label_prefix} MDD: {_fmt_money(dd.mdd)}",
        step=None,
    )

def _fmt_money(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:,.2f}".replace(",", " ")

def _cum_equity(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.cumsum(arr)
def _compute_top_drawdowns(equity: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Находит ТОП-N drawdown эпизодов (не перекрывающиеся):
      peak -> trough -> recovery (до peak или нового high).
    Возвращает список dict:
      {
        "dd": DrawdownInfo(mdd, peak_idx, trough_idx, start_idx, end_idx),
        "pct": float,  # глубина в процентах вниз от peak (по модулю peak)
        "peak_val": float,
        "trough_val": float,
      }
    Сортировка — по mdd (денежная глубина) по убыванию.
    """
    eq = np.asarray(equity, dtype=np.float64)
    n = int(eq.size)
    if n <= 1:
        return []

    events: List[Dict[str, Any]] = []

    peak_idx = 0
    peak_val = float(eq[0])
    i = 1

    while i < n:
        v = float(eq[i])

        # новый пик (>= чтобы "плоский" пик считался пиком)
        if v >= peak_val:
            peak_val = v
            peak_idx = i
            i += 1
            continue

        # начался drawdown эпизод
        trough_idx = i
        trough_val = v
        i += 1

        # ищем дно, пока не восстановились до peak_val
        while i < n and float(eq[i]) < peak_val:
            vv = float(eq[i])
            if vv < trough_val:
                trough_val = vv
                trough_idx = i
            i += 1

        # recovery_idx — точка восстановления (первый индекс где eq >= peak_val),
        # либо конец ряда, если восстановления нет
        recovery_idx = i if i < n else (n - 1)

        mdd = float(peak_val - trough_val)
        if mdd > 0.0 and recovery_idx > peak_idx:
            denom = abs(peak_val)
            pct = (mdd / denom * 100.0) if denom > 1e-12 else 0.0

            dd = DrawdownInfo(
                mdd=mdd,
                peak_idx=int(peak_idx),
                trough_idx=int(trough_idx),
                start_idx=int(peak_idx),
                end_idx=int(recovery_idx),
            )
            events.append(
                {
                    "dd": dd,
                    "pct": float(pct),
                    "peak_val": float(peak_val),
                    "trough_val": float(trough_val),
                }
            )

        # если восстановились — новая база для следующего эпизода
        if i < n:
            peak_idx = i
            peak_val = float(eq[i])
            i += 1

    # ТОП по денежной глубине (как "самые большие просадки" в абсолюте)
    events.sort(key=lambda e: float(e["dd"].mdd), reverse=True)
    return events[: max(0, int(top_n))]


def _annotate_top_drawdowns(
    ax: plt.Axes,
    x: np.ndarray,
    eq: np.ndarray,
    events: List[Dict[str, Any]],
    label_prefix: str,
) -> Tuple[List[Any], List[str]]:
    """
    Рисует drawdown-области для events (в порядке как передали),
    ставит номер у trough, и возвращает (handles, labels) для легенды.
    """
    handles: List[Any] = []
    labels: List[str] = []

    n = len(x)
    for rank, ev in enumerate(events, start=1):
        dd: DrawdownInfo = ev["dd"]
        pct: float = float(ev["pct"])

        if dd.mdd <= 0:
            continue
        if dd.start_idx < 0 or dd.start_idx >= n:
            continue
        if dd.end_idx <= dd.start_idx or dd.end_idx >= n:
            continue

        # label: процент вниз + денежная глубина
        lab = f"{label_prefix} DD#{rank}: -{pct:.2f}% ({_fmt_money(dd.mdd)})"

        seg_x = x[dd.start_idx : dd.end_idx + 1]
        seg_eq = eq[dd.start_idx : dd.end_idx + 1]
        seg_run_max = np.maximum.accumulate(seg_eq)

        # Заливаем область drawdown'а
        h = ax.fill_between(
            seg_x,
            seg_eq,
            seg_run_max,
            alpha=0.20,
            label=lab,
            step=None,
            color="0.5",  # нейтральный серый, чтобы не спорить с линиями equity
        )

        handles.append(h)
        labels.append(lab)

        # Номер у дна
        t = int(dd.trough_idx)
        if 0 <= t < n:
            ax.annotate(
                str(rank),
                (x[t], eq[t]),
                textcoords="offset points",
                xytext=(0, -10),
                ha="center",
                va="top",
                fontsize=8,
            )

    return handles, labels

def _series_basic_stats(s: pd.Series, profit: pd.Series) -> Dict[str, Any]:
    """
    Базовые статистики по серии s и её связи с profit_total.
    Если серия пустая/вся NaN — возвращаются count=0 и остальные значения None.
    """
    s = s.dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "corr_with_profit_total": None,
        }
    stats: Dict[str, Any] = {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "corr_with_profit_total": None,
    }
    # Корреляция с профитом, если есть хотя бы 2 точки
    common_index = s.index.intersection(profit.index)
    if common_index.shape[0] >= 2:
        v1 = s.loc[common_index].to_numpy(dtype=float)
        v2 = profit.loc[common_index].to_numpy(dtype=float)
        if np.std(v1) > 0 and np.std(v2) > 0:
            corr = float(np.corrcoef(v1, v2)[0, 1])
            stats["corr_with_profit_total"] = corr
        else:
            stats["corr_with_profit_total"] = None
    return stats

def _compute_mdd(equity: np.ndarray) -> DrawdownInfo:
    """
    Максимальная просадка: mdd = max(peak - trough), возвращается положительное число.
    Индексы пика и дна — по массиву equity.
    """
    if equity.size == 0:
        return DrawdownInfo(mdd=0.0, peak_idx=0, trough_idx=0, start_idx=0, end_idx=0)

    run_max = np.maximum.accumulate(equity)
    drawdowns = equity - run_max  # <= 0
    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(equity[: trough_idx + 1])) if trough_idx > 0 else 0
    mdd = float(run_max[trough_idx] - equity[trough_idx])  # > 0
    return DrawdownInfo(mdd=mdd, peak_idx=peak_idx, trough_idx=trough_idx, start_idx=peak_idx, end_idx=trough_idx)

def _ensure_dataframe(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Преобразует входной список словарей в DataFrame.

    Новая схема:
      - обязательные поля: open_time, signal, type_of_close, duration_min, profit, fee
      - нет profit_fut / profit_opt
      - profit_total == profit (для совместимости с остальным кодом)

    Если каких-то необязательных полей нет — создаём их (index/squeeze_count/atr/rsi/iv),
    чтобы статистика и графики не падали.
    """
    if not isinstance(trades, list) or len(trades) == 0:
        raise ValueError("Ожидается непустой список словарей со сделками.")

    df = pd.DataFrame(trades).copy()

    # Обязательные поля (новая схема)
    required = [
        "open_time",
        "signal",
        "type_of_close",
        "duration_min",
        "profit",
        "fee",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Отсутствует обязательное поле '{col}' в данных.")

    # Необязательные, но желательные фичи
    for col in ["index", "squeeze_count", "atr", "rsi", "iv"]:
        if col not in df.columns:
            df[col] = np.nan

    # Приведение типов (числовые поля)
    num_cols = [
        "open_time",
        "duration_min",
        "profit",
        "fee",
        "index",
        "squeeze_count",
        "atr",
        "rsi",
        "iv",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Строковые
    df["type_of_close"] = df["type_of_close"].astype(str)

    # Сортировка по времени
    df = df.sort_values("open_time").reset_index(drop=True)

    # Преобразуем в datetime для оси X (Europe/London)
    df["open_dt"] = pd.to_datetime(df["open_time"], unit="s", utc=True).dt.tz_convert("Europe/London")

    # Производные (для совместимости со старым кодом)
    df["profit_total"] = df["profit"]

    # Дни недели (0=Mon..6=Sun) + краткое имя
    df["weekday_idx"] = df["open_dt"].dt.weekday
    df["weekday"] = df["weekday_idx"].map(lambda i: _WEEKDAY_NAMES[int(i)] if pd.notnull(i) else "NA")

    return df


def _weekday_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегация по дням недели с гарантированным порядком Mon..Sun.

    Новая схема:
      - основной PnL: profit
      - метрики:
          count, profit_total
          wins, losses, flats, winrate_pct
          avg_profit_per_trade
          index_mean/index_median
          squeeze_mean/squeeze_median
          atr_mean/atr_median
          rsi_mean/rsi_median
          iv_mean/iv_median

    Если каких-то полей (index/squeeze/atr/rsi/iv) не было в исходных данных,
    соответствующие колонки будут 0.0 после fillna, но логика не упадёт.
    """
    # Базовая таблица дней недели Mon..Sun
    base = pd.DataFrame({"weekday_idx": list(range(7)), "weekday": _WEEKDAY_NAMES})

    if df is None or df.empty:
        cols = [
            "weekday_idx",
            "weekday",
            "count",
            "profit_total",
            "wins",
            "losses",
            "flats",
            "winrate_pct",
            "avg_profit_per_trade",
            "index_mean",
            "index_median",
            "squeeze_mean",
            "squeeze_median",
            "atr_mean",
            "atr_median",
            "rsi_mean",
            "rsi_median",
            "iv_mean",
            "iv_median",
        ]
        return pd.DataFrame(columns=cols)

    # Гарантируем нужные колонки (если где-то забыли создать)
    if "weekday_idx" not in df.columns or "weekday" not in df.columns:
        raise KeyError("Ожидаются колонки 'weekday_idx' и 'weekday' (обычно создаются в _ensure_dataframe).")
    if "profit" not in df.columns:
        raise KeyError("Ожидается колонка 'profit'.")

    g = df.groupby(["weekday_idx", "weekday"], sort=False)

    agg = g.agg(
        count=("profit", "size"),
        profit_total=("profit", "sum"),
        wins=("profit", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        losses=("profit", lambda s: int((pd.to_numeric(s, errors="coerce") < 0).sum())),
        flats=("profit", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
        index_mean=("index", "mean") if "index" in df.columns else ("profit", "mean"),
        index_median=("index", "median") if "index" in df.columns else ("profit", "median"),
        squeeze_mean=("squeeze_count", "mean") if "squeeze_count" in df.columns else ("profit", "mean"),
        squeeze_median=("squeeze_count", "median") if "squeeze_count" in df.columns else ("profit", "median"),
        atr_mean=("atr", "mean") if "atr" in df.columns else ("profit", "mean"),
        atr_median=("atr", "median") if "atr" in df.columns else ("profit", "median"),
        rsi_mean=("rsi", "mean") if "rsi" in df.columns else ("profit", "mean"),
        rsi_median=("rsi", "median") if "rsi" in df.columns else ("profit", "median"),
        iv_mean=("iv", "mean") if "iv" in df.columns else ("profit", "mean"),
        iv_median=("iv", "median") if "iv" in df.columns else ("profit", "median"),
    ).reset_index()

    # Если каких-то фич не было, мы подставили profit-агрегаты выше — это плохо семантически.
    # Поэтому аккуратно зануляем такие колонки, чтобы не вводить в заблуждение.
    if "index" not in df.columns:
        agg["index_mean"] = np.nan
        agg["index_median"] = np.nan
    if "squeeze_count" not in df.columns:
        agg["squeeze_mean"] = np.nan
        agg["squeeze_median"] = np.nan
    if "atr" not in df.columns:
        agg["atr_mean"] = np.nan
        agg["atr_median"] = np.nan
    if "rsi" not in df.columns:
        agg["rsi_mean"] = np.nan
        agg["rsi_median"] = np.nan
    if "iv" not in df.columns:
        agg["iv_mean"] = np.nan
        agg["iv_median"] = np.nan

    # winrate и средний профит на сделку
    agg["winrate_pct"] = agg.apply(lambda r: _percent(r["wins"], r["count"]), axis=1)
    agg["avg_profit_per_trade"] = agg.apply(
        lambda r: (float(r["profit_total"]) / int(r["count"])) if int(r["count"]) else 0.0,
        axis=1,
    )

    # Переупорядочим Mon..Sun, добавим отсутствующие дни
    out = base.merge(agg, on=["weekday_idx", "weekday"], how="left")

    # Для метрик, где логично "0" при отсутствии сделок — ставим 0
    fill_zero_cols = ["count", "profit_total", "wins", "losses", "flats", "winrate_pct", "avg_profit_per_trade"]
    for c in fill_zero_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)

    # Фичи лучше оставить NaN если нет данных
    # (но если у тебя раньше было строго 0.0 — можно заменить на fillna(0.0))
    for c in ["index_mean", "index_median", "squeeze_mean", "squeeze_median", "atr_mean", "atr_median",
              "rsi_mean", "rsi_median", "iv_mean", "iv_median"]:
        if c in out.columns:
            # если день есть, но сделок нет — оставим NaN, так честнее
            pass

    # Приведение типов для целочисленных колонок
    for col in ["count", "wins", "losses", "flats"]:
        out[col] = out[col].astype(int)

    return out


def _percent(n: float, d: float) -> float:
    return 0.0 if d == 0 else 100.0 * n / d

def compute_trade_stats(trades: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Счёт статистики без построения графиков.
    Возвращает (stats_dict, df) — второй элемент можно передать в построитель графиков.

    Новая схема:
      - основная метрика PnL: df["profit"]
      - нет profit_fut / profit_opt
      - profit_total == profit (оставлено для совместимости)
    """
    df = _ensure_dataframe(trades)

    n_trades = int(len(df))

    # В новой схеме опционов отдельно нет; ключ оставляем ради совместимости
    has_opt = False

    # Сводки по типу закрытия
    by_close = df.groupby("type_of_close").agg(
        count=("profit", "size"),
        profit_total=("profit", "sum"),
        index_mean=("index", "mean"),
        index_median=("index", "median"),
        squeeze_mean=("squeeze_count", "mean"),
        squeeze_median=("squeeze_count", "median"),
        atr_mean=("atr", "mean"),
        atr_median=("atr", "median"),
        rsi_mean=("rsi", "mean"),
        rsi_median=("rsi", "median"),
        iv_mean=("iv", "mean"),
        iv_median=("iv", "median"),
    ).sort_values("profit_total", ascending=False)

    # Сводки по сигналам
    by_signal = df.groupby("signal").agg(
        count=("profit", "size"),
        profit_total=("profit", "sum"),
        index_mean=("index", "mean"),
        index_median=("index", "median"),
        squeeze_mean=("squeeze_count", "mean"),
        squeeze_median=("squeeze_count", "median"),
        atr_mean=("atr", "mean"),
        atr_median=("atr", "median"),
        rsi_mean=("rsi", "mean"),
        rsi_median=("rsi", "median"),
        iv_mean=("iv", "mean"),
        iv_median=("iv", "median"),
    ).sort_index()

    by_weekday_df = _weekday_agg(df)

    # Тоталы
    profit_total = float(df["profit"].sum())

    # Кум. доходности и просадки (по одной equity-кривой)
    eq = _cum_equity(df["profit"].to_numpy(dtype=float))
    dd_total = _compute_mdd(eq)

    # Доп. метрики
    wins = int((df["profit"] > 0).sum())
    losses = int((df["profit"] < 0).sum())
    flats = n_trades - wins - losses
    winrate = _percent(wins, n_trades)

    # Статистика по index / squeeze_count / atr / rsi / iv
    index_stats = _series_basic_stats(df["index"], df["profit"])
    squeeze_stats = _series_basic_stats(df["squeeze_count"], df["profit"])
    atr_stats = _series_basic_stats(df["atr"], df["profit"])
    rsi_stats = _series_basic_stats(df["rsi"], df["profit"])
    iv_stats = _series_basic_stats(df["iv"], df["profit"])

    stats: Dict[str, Any] = {
        "n_trades": n_trades,
        "has_opt": has_opt,

        # totals: оставляем старые ключи ради совместимости, но фактически всё = profit_total
        "totals": {
            "profit_total": profit_total,
            "profit_fut_total": profit_total,  # backward-compat
            "profit_opt_total": 0.0,            # backward-compat
        },

        "by_type_of_close": by_close.reset_index().to_dict(orient="records"),
        "by_signal": by_signal.reset_index().to_dict(orient="records"),
        "by_weekday": by_weekday_df.to_dict(orient="records"),

        # drawdowns: оставляем структуру, но считаем только total
        "drawdowns": {
            "fut": {
                "mdd": dd_total.mdd,
                "start_idx": dd_total.start_idx,
                "end_idx": dd_total.end_idx,
            },  # backward-compat
            "opt": None,  # нет отдельной опционной части
            "total": {
                "mdd": dd_total.mdd,
                "start_idx": dd_total.start_idx,
                "end_idx": dd_total.end_idx,
            },
        },

        "outcomes": {
            "wins": wins,
            "losses": losses,
            "flats": flats,
            "winrate_pct": winrate,
            "avg_profit_per_trade": profit_total / n_trades if n_trades else 0.0,
        },

        "index_stats": index_stats,
        "squeeze_stats": squeeze_stats,
        "atr_stats": atr_stats,
        "rsi_stats": rsi_stats,
        "iv_stats": iv_stats,
    }

    return stats, df



#========================================
#########FUNCTIONS PLOT##################
#========================================

# def _plot_distributions(
#     df: pd.DataFrame,
#     stats: Dict[str, Any],
#     show: bool,
#     save: bool,
#     save_dir: Optional[str],
#     dpi: int,
# ) -> None:
#     # Подготовка таблиц
#     df_type = pd.DataFrame(stats["by_type_of_close"])
#     df_sig = pd.DataFrame(stats["by_signal"])

#     # Фигура 2: 2x2 — количество и прибыль по типам закрытия и сигналам
#     fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
#     (ax1, ax2), (ax3, ax4) = axs

#     # 1) Количество по type_of_close (столбики)
#     if not df_type.empty:
#         ax1.bar(df_type["type_of_close"], df_type["count"])
#         ax1.set_title("Trades by type_of_close (count)")
#         ax1.set_ylabel("Count")
#         ax1.grid(True, axis="y", alpha=0.25)
#         for tick in ax1.get_xticklabels():
#             tick.set_rotation(20)
#     else:
#         ax1.text(0.5, 0.5, "No data", ha="center", va="center")

#     # 2) Прибыль по type_of_close (столбики total)
#     if not df_type.empty:
#         ax2.bar(df_type["type_of_close"], df_type["profit_total"])
#         ax2.set_title("Profit by type_of_close (total)")
#         ax2.set_ylabel("Profit")
#         ax2.grid(True, axis="y", alpha=0.25)
#         for tick in ax2.get_xticklabels():
#             tick.set_rotation(20)
#     else:
#         ax2.text(0.5, 0.5, "No data", ha="center", va="center")

#     # 3) Доли по сигналам (круг)
#     if not df_sig.empty:
#         ax3.pie(
#             df_sig["count"],
#             labels=list(map(str, df_sig["signal"])),
#             autopct="%1.1f%%",
#             startangle=90,
#         )
#         ax3.set_title("Signals share (count)")
#     else:
#         ax3.text(0.5, 0.5, "No data", ha="center", va="center")

#     # 4) Прибыль по сигналам (столбики)
#     if not df_sig.empty:
#         ax4.bar(list(map(str, df_sig["signal"])), df_sig["profit_total"])
#         ax4.set_title("Profit by signal (total)")
#         ax4.set_ylabel("Profit")
#         ax4.grid(True, axis="y", alpha=0.25)
#     else:
#         ax4.text(0.5, 0.5, "No data", ha="center", va="center")

#     plt.tight_layout()
#     _maybe_save_or_show(fig, "stats_overview.png", show, save, save_dir, dpi)

def _plot_equity(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    title: Optional[str],
    show: bool,
    save: bool,
    save_dir: Optional[str],
    dpi: int,
) -> None:
    """
    Equity-график (1 кривая) + ТОП-10 drawdowns + столбики profit по сделкам.
    Требуемые колонки df:
      - open_dt (datetime, можно timezone-aware)
      - profit (float)
      - duration_min (float)  (для статистики; если нет — будет "—")
    """
    if df is None or len(df) == 0:
        return
    if "open_dt" not in df.columns:
        return
    if "profit" not in df.columns:
        return

    # --- подготовка времени ---
    x_dt = pd.to_datetime(df["open_dt"], errors="coerce")
    if x_dt.isna().all():
        return

    # matplotlib лучше работает с naive datetime; приводим к UTC и убираем tz
    try:
        if getattr(x_dt.dt, "tz", None) is not None:
            x_dt = x_dt.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # если что-то нестандартное — просто делаем naive
        try:
            x_dt = x_dt.dt.tz_localize(None)
        except Exception:
            pass

    # --- profit ---
    profit = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # сортировка по времени (на всякий случай)
    order = np.argsort(x_dt.to_numpy())
    x_dt = x_dt.iloc[order].reset_index(drop=True)
    profit = profit[order]

    x_num = mdates.date2num(x_dt.to_numpy())

    # --- equity (накопление profit) ---
    eq = _cum_equity(profit)

    # --- ширина столбиков (авто под шаг времени) ---
    # берём уникальные x, считаем положительные разницы, берём "типичный" шаг
    uniq = np.unique(x_num)
    d = np.diff(uniq)
    d = d[d > 0]
    if d.size > 0:
        typical_step = float(np.median(d))
        bar_w = max(typical_step * 0.85, 0.0004)  # минимум ~0.6 минуты в днях
        bar_w = min(bar_w, 2.0)                   # максимум 2 дня, чтобы не раздувало редкие точки
    else:
        bar_w = 0.02  # fallback (~30 минут)

    # --- базовая статистика (текстом) ---
    n_trades = int(len(profit))
    total_balance = float(np.nansum(profit))

    def _fmt_minutes(m: Optional[float]) -> str:
        if m is None or (isinstance(m, float) and (math.isnan(m) or math.isinf(m))):
            return "—"
        mm = float(m)
        if mm < 0:
            return "—"
        # показываем как "X min" или "Y h Z min"
        h = int(mm // 60)
        r = int(round(mm - h * 60))
        if h <= 0:
            return f"{int(round(mm))} min"
        return f"{h} h {r} min"

    dur_all = None
    dur_win = None
    dur_loss = None
    if "duration_min" in df.columns:
        dur = pd.to_numeric(df["duration_min"], errors="coerce").to_numpy(dtype=float)[order]
        m_all = np.isfinite(dur)
        if np.any(m_all):
            dur_all = float(np.nanmean(dur[m_all]))

        m_win = np.isfinite(dur) & (profit > 0)
        if np.any(m_win):
            dur_win = float(np.nanmean(dur[m_win]))

        m_loss = np.isfinite(dur) & (profit < 0)
        if np.any(m_loss):
            dur_loss = float(np.nanmean(dur[m_loss]))

    # --- figure layout ---
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0]},
    )
    ax_top, ax_bottom = axes

    # --- верх: equity ---
    ax_top.plot(x_num, eq, linewidth=1.8, label="Equity")
    ax_top.set_ylabel("Equity")
    ax_top.grid(True, alpha=0.35)

    if title:
        ax_top.set_title(title, pad=10)

    # легенда по линии equity
    line_legend = ax_top.legend(loc="best")
    ax_top.add_artist(line_legend)

    # --- ТОП-10 просадок ---
    top_dd = _compute_top_drawdowns(eq, top_n=10)
    dd_handles, dd_labels = _annotate_top_drawdowns(
        ax=ax_top,
        x=x_num,
        eq=eq,
        events=top_dd,
        label_prefix="EQ",
    )

    if dd_handles:
        ax_top.legend(
            dd_handles,
            dd_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            title=f"TOP-{len(dd_handles)} drawdowns",
        )
        fig.subplots_adjust(right=0.78)

    # --- блок статистики сверху (слева) ---
    stat_text = (
        f"Trades: {n_trades}\n"
        f"Total balance: {_fmt_money(total_balance)}\n"
        f"Avg duration (all): {_fmt_minutes(dur_all)}\n"
        f"Avg duration (profit): {_fmt_minutes(dur_win)}\n"
        f"Avg duration (loss): {_fmt_minutes(dur_loss)}"
    )
    ax_top.text(
        0.01, 0.99,
        stat_text,
        transform=ax_top.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.15",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, edgecolor="0.7"),
    )

    # --- низ: profit bars ---
    # чтобы столбики читались, красим по знаку
    bar_colors = np.where(profit >= 0, "limegreen", "crimson")
    ax_bottom.bar(x_num, profit, width=bar_w, align="center", color=bar_colors, linewidth=0.0)
    ax_bottom.axhline(0.0, linewidth=1.0)
    ax_bottom.set_ylabel("Profit")
    ax_bottom.grid(True, axis="y", alpha=0.35)

    # формат оси X
    ax_bottom.xaxis_date()
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=20)

    _maybe_save_or_show(fig, "equity.png", show, save, save_dir, dpi)

def _plot_distributions(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    show: bool,
    save: bool,
    save_dir: Optional[str],
    dpi: int,
) -> None:
    # Подготовка таблиц
    df_type = pd.DataFrame(stats["by_type_of_close"])
    df_sig = pd.DataFrame(stats["by_signal"])

    # Фигура 2: 2x2 — количество и прибыль по типам закрытия и сигналам
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    (ax1, ax2), (ax3, ax4) = axs

    # 1) Количество по type_of_close (столбики)
    if not df_type.empty:
        ax1.bar(df_type["type_of_close"], df_type["count"])
        ax1.set_title("Trades by type_of_close (count)")
        ax1.set_ylabel("Count")
        ax1.grid(True, axis="y", alpha=0.25)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(20)
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center")

    # 2) Прибыль по type_of_close (столбики total)
    if not df_type.empty:
        ax2.bar(df_type["type_of_close"], df_type["profit_total"])
        ax2.set_title("Profit by type_of_close (total)")
        ax2.set_ylabel("Profit")
        ax2.grid(True, axis="y", alpha=0.25)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(20)
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center")

    # --- ДОБАВЛЕНО: сводка по сделкам profit<0 / profit>0 + средние ---
    profit_note = None
    if (df is not None) and (not df.empty) and ("profit" in df.columns):
        pr = pd.to_numeric(df["profit"], errors="coerce").dropna()
        if not pr.empty:
            neg = pr[pr < 0]
            pos = pr[pr > 0]

            cnt_neg = int(neg.shape[0])
            cnt_pos = int(pos.shape[0])

            mean_neg = float(neg.mean()) if cnt_neg > 0 else float("nan")
            mean_pos = float(pos.mean()) if cnt_pos > 0 else float("nan")

            def _fmt(x: float) -> str:
                return "n/a" if pd.isna(x) else f"{x:.6g}"

            profit_note = (
                f"Profit>0 trades: {cnt_pos} (avg {_fmt(mean_pos)})\n"
                f"Profit<0 trades: {cnt_neg} (avg {_fmt(mean_neg)})"
            )

    if profit_note:
        ax2.text(
            0.02,
            0.98,
            profit_note,
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    # --- /ДОБАВЛЕНО ---

    # 3) и 4) Сигналы:
    # Если df_sig содержит колонку dow, то рисуем по дням недели
    # и ДЕЛАЕМ по ДВА столбика на каждый день: signal=1 и signal=2.
    # Иначе — поведение как раньше: pie по count и bar по profit_total по сигналам.
    has_dow = (not df_sig.empty) and ("dow" in df_sig.columns)

    if has_dow:
        dow_order = [0, 1, 2, 3, 4, 5, 6]
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        df_sig_loc = df_sig.copy()
        df_sig_loc["dow"] = pd.to_numeric(df_sig_loc["dow"], errors="coerce")
        df_sig_loc["signal"] = pd.to_numeric(df_sig_loc["signal"], errors="coerce")

        # оставляем только сигналы 1 и 2
        df_sig_loc = df_sig_loc[df_sig_loc["signal"].isin([1, 2])].copy()

        # --- ax3: grouped bars by dow (count) ---
        if (not df_sig_loc.empty) and ("count" in df_sig_loc.columns):
            pv_cnt = (
                df_sig_loc.pivot_table(index="dow", columns="signal", values="count", aggfunc="sum")
                .reindex(dow_order)
                .fillna(0.0)
            )

            x = list(range(len(dow_order)))
            width = 0.38
            x1 = [xi - width / 2.0 for xi in x]
            x2 = [xi + width / 2.0 for xi in x]

            y1 = pv_cnt[1].astype(float).tolist() if 1 in pv_cnt.columns else [0.0] * len(dow_order)
            y2 = pv_cnt[2].astype(float).tolist() if 2 in pv_cnt.columns else [0.0] * len(dow_order)

            ax3.bar(x1, y1, width=width, label="signal=1")
            ax3.bar(x2, y2, width=width, label="signal=2")
            ax3.set_title("Signals by day-of-week (count)")
            ax3.set_ylabel("Count")
            ax3.set_xticks(x)
            ax3.set_xticklabels(dow_labels, rotation=0)
            ax3.grid(True, axis="y", alpha=0.25)
            ax3.legend(loc="best", fontsize=9)
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center")

        # --- ax4: grouped bars by dow (profit_total) ---
        if (not df_sig_loc.empty) and ("profit_total" in df_sig_loc.columns):
            pv_p = (
                df_sig_loc.pivot_table(index="dow", columns="signal", values="profit_total", aggfunc="sum")
                .reindex(dow_order)
                .fillna(0.0)
            )

            x = list(range(len(dow_order)))
            width = 0.38
            x1 = [xi - width / 2.0 for xi in x]
            x2 = [xi + width / 2.0 for xi in x]

            y1 = pv_p[1].astype(float).tolist() if 1 in pv_p.columns else [0.0] * len(dow_order)
            y2 = pv_p[2].astype(float).tolist() if 2 in pv_p.columns else [0.0] * len(dow_order)

            ax4.bar(x1, y1, width=width, label="signal=1")
            ax4.bar(x2, y2, width=width, label="signal=2")
            ax4.set_title("Signals by day-of-week (profit_total)")
            ax4.set_ylabel("Profit")
            ax4.set_xticks(x)
            ax4.set_xticklabels(dow_labels, rotation=0)
            ax4.grid(True, axis="y", alpha=0.25)
            ax4.legend(loc="best", fontsize=9)
        else:
            ax4.text(0.5, 0.5, "No data", ha="center", va="center")

    else:
        # 3) Доли по сигналам (круг) — как было
        if not df_sig.empty:
            ax3.pie(
                df_sig["count"],
                labels=list(map(str, df_sig["signal"])),
                autopct="%1.1f%%",
                startangle=90,
            )
            ax3.set_title("Signals share (count)")
        else:
            ax3.text(0.5, 0.5, "No data", ha="center", va="center")

        # 4) Прибыль по сигналам (столбики) — как было
        if not df_sig.empty:
            ax4.bar(list(map(str, df_sig["signal"])), df_sig["profit_total"])
            ax4.set_title("Profit by signal (total)")
            ax4.set_ylabel("Profit")
            ax4.grid(True, axis="y", alpha=0.25)
        else:
            ax4.text(0.5, 0.5, "No data", ha="center", va="center")

    plt.tight_layout()
    _maybe_save_or_show(fig, "stats_overview.png", show, save, save_dir, dpi)




def _plot_weekday_stats(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    show: bool,
    save: bool,
    save_dir: Optional[str],
    dpi: int,
) -> None:
    if df is None or len(df) == 0:
        return
    if "profit" not in df.columns or "signal" not in df.columns:
        return

    df_loc = df.copy()

    # --- weekday: берём df["dow"] или вычисляем из open_dt ---
    if "dow" in df_loc.columns:
        dow = pd.to_numeric(df_loc["dow"], errors="coerce")
    elif "open_dt" in df_loc.columns:
        dt = pd.to_datetime(df_loc["open_dt"], errors="coerce")
        if dt.isna().all():
            return
        dow = dt.dt.dayofweek  # Monday=0 ... Sunday=6
    else:
        return

    df_loc["dow"] = pd.to_numeric(dow, errors="coerce")
    df_loc["signal"] = pd.to_numeric(df_loc["signal"], errors="coerce")
    df_loc["profit"] = pd.to_numeric(df_loc["profit"], errors="coerce").fillna(0.0)

    # нормализуем: сигнал в int, dow в int
    df_loc = df_loc[df_loc["dow"].notna() & df_loc["signal"].notna()].copy()
    if len(df_loc) == 0:
        return

    df_loc["dow"] = df_loc["dow"].round().astype(int)
    df_loc["signal"] = df_loc["signal"].round().astype(int)

    # только дни 0..6 и сигналы 1..2
    df_loc = df_loc[df_loc["dow"].isin([0, 1, 2, 3, 4, 5, 6]) & df_loc["signal"].isin([1, 2])].copy()
    if len(df_loc) == 0:
        return

    # --- агрегаты по (dow, signal) ---
    df_loc["is_win"] = (df_loc["profit"] > 0).astype(int)

    agg = (
        df_loc.groupby(["dow", "signal"], as_index=False)
        .agg(
            count=("profit", "size"),
            profit_total=("profit", "sum"),
            wins=("is_win", "sum"),
        )
    )

    # avg profit per trade
    agg["avg_profit_per_trade"] = agg["profit_total"] / agg["count"].replace(0, np.nan)
    agg["avg_profit_per_trade"] = agg["avg_profit_per_trade"].fillna(0.0)

    # winrate
    agg["winrate_pct"] = (agg["wins"] / agg["count"].replace(0, np.nan) * 100.0).fillna(0.0)

    # --- pivots (гарантируем 2 сигнала всегда) ---
    dow_order = [0, 1, 2, 3, 4, 5, 6]
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    def _pivot(metric: str) -> pd.DataFrame:
        pv = (
            agg.pivot_table(index="dow", columns="signal", values=metric, aggfunc="sum")
            .reindex(dow_order)
            .fillna(0.0)
        )
        if 1 not in pv.columns:
            pv[1] = 0.0
        if 2 not in pv.columns:
            pv[2] = 0.0
        pv = pv[[1, 2]]
        return pv

    pv_cnt = _pivot("count")
    pv_profit = _pivot("profit_total")
    pv_avg = _pivot("avg_profit_per_trade")
    pv_wr = _pivot("winrate_pct")

    x = np.arange(len(dow_order), dtype=float)
    width = 0.38

    # ОДНА фигура 2x2: Count, Total Profit, Avg Profit, Winrate (всё на одной картинке)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 9.2))
    (ax1, ax2), (ax3, ax4) = axs

    # 1) Count
    ax1.bar(x - width / 2.0, pv_cnt[1].to_numpy(dtype=float), width=width, label="signal=1")
    ax1.bar(x + width / 2.0, pv_cnt[2].to_numpy(dtype=float), width=width, label="signal=2")
    ax1.set_title("Trades by weekday (count)")
    ax1.set_ylabel("Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dow_labels, rotation=0)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best", fontsize=9)

    # 2) Total profit
    ax2.bar(x - width / 2.0, pv_profit[1].to_numpy(dtype=float), width=width, label="signal=1")
    ax2.bar(x + width / 2.0, pv_profit[2].to_numpy(dtype=float), width=width, label="signal=2")
    ax2.set_title("Profit by weekday (total)")
    ax2.set_ylabel("Profit")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dow_labels, rotation=0)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="best", fontsize=9)

    # 3) Avg profit per trade
    ax3.bar(x - width / 2.0, pv_avg[1].to_numpy(dtype=float), width=width, label="signal=1")
    ax3.bar(x + width / 2.0, pv_avg[2].to_numpy(dtype=float), width=width, label="signal=2")
    ax3.set_title("Avg profit per trade by weekday")
    ax3.set_ylabel("Avg profit")
    ax3.set_xticks(x)
    ax3.set_xticklabels(dow_labels, rotation=0)
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(loc="best", fontsize=9)

    # 4) Winrate
    ax4.bar(x - width / 2.0, pv_wr[1].to_numpy(dtype=float), width=width, label="signal=1")
    ax4.bar(x + width / 2.0, pv_wr[2].to_numpy(dtype=float), width=width, label="signal=2")
    ax4.set_title("Winrate by weekday")
    ax4.set_ylabel("Winrate, %")
    ax4.set_xticks(x)
    ax4.set_xticklabels(dow_labels, rotation=0)
    ax4.grid(True, axis="y", alpha=0.25)
    ax4.legend(loc="best", fontsize=9)

    plt.tight_layout()
    _maybe_save_or_show(fig, "weekday_stats.png", show, save, save_dir, dpi)




def _plot_trades_timeline(
    df: pd.DataFrame,
    stats: Dict[str, Any],
    *,
    show: bool = True,
    save: bool = False,
    save_dir: Optional[str] = None,
    dpi: int = 140,
    title: Optional[str] = None,
) -> None:
    """
    Тайм-график сделок (PNG на каждый год; при необходимости год делится на H1/H2):
      - каждая сделка = столбик (height = profit_total)
      - цвет сделки = signal (1 или 2)
          signal=1 -> blueviolet
          signal=2 -> darkorange
      - фон недель: зелёная штриховка если сальдо недели > 0, красная если < 0
      - подпись сальдо недели: мелко, вертикально
      - события: если в день сделки есть события из events_2020_2026_ai.csv,
                пишем (очень мелко, вертикально): event_type sentiment ai_sentiment
                ПРАВИЛО (у линии нуля):
                  * если столбик вверх (profit >= 0) — подпись НИЖЕ линии нуля
                  * если столбик вниз (profit < 0)  — подпись ВЫШЕ линии нуля
      - ширина столбика = 1 день
      - левый край столбика совпадает с границей дня (и недельная линия понедельника совпадает с левым краем понедельника)
    """
    if df is None or len(df) == 0:
        return

    # =============================================================================
    # EVENTS (root: events_2020_2026_ai.csv)
    # =============================================================================
    EVENTS_CSV_PATH = "events_2020_2026_ai.csv"

    def _load_events_map(csv_path: str) -> Dict[object, List[str]]:
        """
        dict: date -> list[str label]
        date: datetime.date (UTC)
        """
        events_map: Dict[object, List[str]] = {}
        try:
            if not os.path.exists(csv_path):
                return events_map

            ev = pd.read_csv(csv_path)
            if "event_time_utc" not in ev.columns or "event_type" not in ev.columns:
                return events_map

            ev_dt = pd.to_datetime(ev["event_time_utc"], utc=True, errors="coerce")
            ev = ev.loc[~ev_dt.isna()].copy()
            if len(ev) == 0:
                return events_map

            # дата в UTC, без tz
            ev_dt = ev_dt.loc[ev.index].dt.tz_convert("UTC").dt.tz_localize(None)
            ev["_date"] = ev_dt.dt.date

            if "sentiment" not in ev.columns:
                ev["sentiment"] = ""
            if "ai_sentiment" not in ev.columns:
                ev["ai_sentiment"] = ""

            for _, row in ev.iterrows():
                d = row["_date"]
                et = str(row.get("event_type", "")).strip()
                s1 = str(row.get("sentiment", "")).strip()
                s2 = str(row.get("ai_sentiment", "")).strip()

                label = f"{et} {s1} {s2}".strip()
                if not label:
                    continue

                events_map.setdefault(d, []).append(label)

            return events_map
        except Exception:
            return {}

    events_map = _load_events_map(EVENTS_CSV_PATH)

    # =============================================================================
    # TIME + PROFIT + SIGNAL
    # =============================================================================
    # берём время в UTC (naive), чтобы недели/дни были стабильны
    if "open_time" in df.columns and pd.api.types.is_numeric_dtype(df["open_time"]):
        dt_utc = pd.to_datetime(df["open_time"].astype(float), unit="s", utc=True, errors="coerce")
    elif "open_dt" in df.columns:
        # open_dt у тебя tz-aware Europe/London, переведём в UTC
        dt_utc = pd.to_datetime(df["open_dt"], utc=True, errors="coerce")
    else:
        return

    dt_utc = dt_utc.dropna()
    if len(dt_utc) == 0:
        return

    df2 = df.loc[dt_utc.index].copy()
    dt_utc = dt_utc.dt.tz_convert("UTC").dt.tz_localize(None)

    # profit
    if "profit_total" in df2.columns:
        y = pd.to_numeric(df2["profit_total"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_label = "Profit (TOTAL)"
    elif "profit" in df2.columns:
        y = pd.to_numeric(df2["profit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y_label = "Profit"
    else:
        fut = pd.to_numeric(df2.get("profit_fut", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        opt = pd.to_numeric(df2.get("profit_opt", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = fut + opt
        y_label = "Profit (FUT+OPT)"

    # signal
    if "signal" in df2.columns:
        sig = pd.to_numeric(df2["signal"], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        sig = np.zeros(len(df2), dtype=int)

    # собираем рабочий dfp
    dfp = df2.copy()
    dfp["_dt"] = dt_utc.to_numpy()
    dfp["_y"] = y
    dfp["_sig"] = sig
    dfp = dfp.sort_values("_dt", kind="mergesort")

    # =============================================================================
    # CONFIG (colors, sizes)
    # =============================================================================
    sig1_color = "blueviolet"
    sig2_color = "darkorange"
    unknown_color = "0.35"

    pos_face = "limegreen"
    pos_edge = "green"
    neg_face = "crimson"
    neg_edge = "red"

    bar_width_days = 1.0
    ev_font = 3.5

    def _make_event_label(labels: List[str]) -> str:
        if not labels:
            return ""
        if len(labels) == 1:
            return labels[0]
        if len(labels) <= 3:
            return " ; ".join(labels)
        return " ; ".join(labels[:3]) + f" ; +{len(labels) - 3}"

    def _plot_range(
        dfr: pd.DataFrame,
        *,
        suffix: str,
        plot_title: str,
    ) -> None:
        if dfr is None or len(dfr) == 0:
            return

        dt_series = pd.to_datetime(dfr["_dt"], errors="coerce")
        dt_series = dt_series.dropna()
        if len(dt_series) == 0:
            return

        y_part = pd.to_numeric(dfr["_y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        sig_part = pd.to_numeric(dfr["_sig"], errors="coerce").fillna(0).astype(int).to_numpy()

        # X: начало дня (левый край столбика)
        day_start = dt_series.dt.normalize()
        x_left = mdates.date2num(day_start.to_numpy())

        # цвет столбиков
        bar_colors = []
        for s in sig_part:
            if int(s) == 1:
                bar_colors.append(sig1_color)
            elif int(s) == 2:
                bar_colors.append(sig2_color)
            else:
                bar_colors.append(unknown_color)

        # недельные суммы: понедельник как старт недели
        week_start = (day_start - pd.to_timedelta(day_start.dt.weekday, unit="D")).dt.normalize()
        weekly_sums = pd.Series(y_part, index=week_start).groupby(level=0).sum()

        ws_min = week_start.min()
        ws_max = week_start.max()
        week_starts_all = pd.date_range(ws_min, ws_max + pd.Timedelta(days=7), freq="7D") if pd.notna(ws_min) else pd.DatetimeIndex([])

        # ----- figure -----
        fig, ax = plt.subplots(figsize=(13.5, 4.4), dpi=dpi)
        ax.set_axisbelow(True)

        # фон недель + линии недель
        if len(week_starts_all) > 0:
            for ws in week_starts_all:
                we = ws + pd.Timedelta(days=7)
                x0 = mdates.date2num(pd.Timestamp(ws).to_pydatetime())
                x1 = mdates.date2num(pd.Timestamp(we).to_pydatetime())

                ax.axvline(x0, linewidth=0.8, color="0.85", zorder=0)

                s_week = float(weekly_sums.get(ws, 0.0))
                if s_week > 0:
                    ax.axvspan(
                        x0, x1,
                        facecolor=pos_face,
                        alpha=0.14,
                        hatch="///",
                        edgecolor=pos_edge,
                        linewidth=0.0,
                        zorder=0,
                    )
                elif s_week < 0:
                    ax.axvspan(
                        x0, x1,
                        facecolor=neg_face,
                        alpha=0.12,
                        hatch="///",
                        edgecolor=neg_edge,
                        linewidth=0.0,
                        zorder=0,
                    )

            # последняя граница (правый край)
            ax.axvline(
                mdates.date2num((week_starts_all[-1] + pd.Timedelta(days=7)).to_pydatetime()),
                linewidth=0.8,
                color="0.85",
                zorder=0,
            )

        # ось 0
        ax.axhline(0.0, linewidth=1.0, color="0.2", zorder=1)

        # бары: align edge, x = левый край дня
        ax.bar(x_left, y_part, width=bar_width_days, color=bar_colors, align="edge", zorder=2)

        # оси/заголовки
        ax.set_ylabel(y_label)
        ax.set_xlabel("Date (UTC)")
        ax.set_title(plot_title)

        # лимиты по Y так, чтобы возле нуля было место для ивентов
        y_min = float(np.min(y_part)) if len(y_part) else 0.0
        y_max = float(np.max(y_part)) if len(y_part) else 0.0
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
        yr = (y_max - y_min)
        pad = (0.10 * yr) if yr > 0 else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

        # формат даты: помесячно (чтобы не засорять)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
        ax.grid(True, axis="y", alpha=0.25)

        # ----- события у линии нуля -----
        if events_map:
            y0, y1 = ax.get_ylim()
            yr2 = (y1 - y0)
            y_off = (yr2 * 0.018) if yr2 != 0 else 0.2

            trade_dates = day_start.dt.date.to_numpy()

            for i in range(len(x_left)):
                d = trade_dates[i]
                labels = events_map.get(d)
                if not labels:
                    continue

                lab = _make_event_label(labels)
                if not lab:
                    continue

                yv = float(y_part[i])

                # X центр столбика: left + 0.5*width
                xt = float(x_left[i]) + (bar_width_days * 0.5)

                # правило от линии нуля
                if yv >= 0:
                    yt = 0.0 - y_off
                    va = "top"
                else:
                    yt = 0.0 + y_off
                    va = "bottom"

                ax.text(
                    xt,
                    yt,
                    lab,
                    fontsize=ev_font,
                    ha="center",
                    va=va,
                    rotation=90,
                    color="0.15",
                    zorder=4,
                    clip_on=True,
                )

        # ----- подписи недельных сумм -----
        if len(week_starts_all) > 0:
            y0, y1 = ax.get_ylim()
            yr2 = (y1 - y0)
            y_text = y0 + (yr2 * 0.02 if yr2 != 0 else 0.2)

            for ws in week_starts_all:
                s_week = float(weekly_sums.get(ws, 0.0))
                if abs(s_week) < 1e-12:
                    continue

                we = ws + pd.Timedelta(days=7)
                x0 = mdates.date2num(pd.Timestamp(ws).to_pydatetime())
                x1 = mdates.date2num(pd.Timestamp(we).to_pydatetime())
                xm = (x0 + x1) * 0.5

                ax.text(
                    xm,
                    y_text,
                    f"Σ {s_week:+.1f}",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    color="0.20",
                    zorder=3,
                )

        # легенда по сигналам
        handles = [
            mpatches.Patch(color=sig1_color, label="signal=1"),
            mpatches.Patch(color=sig2_color, label="signal=2"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2, frameon=True)

        fig.tight_layout()

        # сохраняем/показываем
        fname = f"trades_timeline_{suffix}.png"
        _maybe_save_or_show(fig, fname, show, save, save_dir, dpi)

    # =============================================================================
    # PER-YEAR PLOTTING
    # =============================================================================
    years = pd.to_datetime(dfp["_dt"], errors="coerce").dt.year.dropna().astype(int).unique()
    years = sorted(list(years))
    if not years:
        return

    base_title = title or "Trades timeline (bars coloured by signal)"

    for yyy in years:
        d_y = dfp[pd.to_datetime(dfp["_dt"], errors="coerce").dt.year == int(yyy)].copy()
        if len(d_y) == 0:
            continue

        _plot_range(
            d_y,
            suffix=f"{int(yyy)}",
            plot_title=f"{base_title} — {int(yyy)}",
        )

def _plot_nested_features_profit_maps(
    trades: List[Dict[str, Any]],
    *,
    show: bool = True,
    save: bool = False,
    save_dir: Optional[str] = None,
    dpi: int = 140,
    gridsize: int = 45,
    signals: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = None,
    debug: bool = False,
) -> None:
    """
    Визуализирует hexbin "feature_value vs profit" для ВСЕХ значений из trades[i]['features'].

    Для каждого feature_name строится figure:
      - 2 subplot'а: отдельно signal=1 и signal=2 (или signals=...)
      - X: значение feature
      - Y: profit
      - сверху: Σprofit по X-бинам
      - плотная нижняя шкала X + вертикальные полупрозрачные линии по ГРАНИЦАМ БИНОВ

    Параметры:
      trades: list[dict] - список сделок
      signals: какие сигналы рисовать (по умолчанию (1,2))
      max_features: ограничить число фич, если их много (None = все)
      debug: печатать диагностику
    """
    if not trades:
        return

    df = pd.DataFrame(trades)
    if df is None or len(df) == 0:
        return

    if "features" not in df.columns:
        return

    # ---------- helpers ----------
    def _coerce_float(v):
        if v is None:
            return np.nan
        try:
            if isinstance(v, np.generic):
                return float(v)
        except Exception:
            pass
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
            v = v[0]
            try:
                if isinstance(v, np.generic):
                    return float(v)
            except Exception:
                pass
        try:
            return float(v)
        except Exception:
            return np.nan

    def _dense_x_axis(ax, *, bin_edges: Optional[np.ndarray] = None) -> None:
        # тики оставляем плотными как раньше
        ax.xaxis.set_major_locator(MaxNLocator(nbins=30, min_n_ticks=16))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="x", which="major", labelsize=5, pad=0, length=3)
        ax.tick_params(axis="x", which="minor", length=2, width=0.8)

        for lab in ax.get_xticklabels(which="major"):
            lab.set_rotation(90)
            lab.set_horizontalalignment("center")
            lab.set_verticalalignment("top")
            lab.set_rotation_mode("anchor")

        ax.margins(x=0)

        # ВАЖНО: вертикальные линии рисуем по ГРАНИЦАМ БИНОВ (если они есть),
        # чтобы линии совпадали с диапазонами столбиков сверху.
        if bin_edges is not None and len(bin_edges) >= 2:
            for xedge in bin_edges:
                ax.axvline(float(xedge), alpha=0.18, linewidth=0.7, zorder=5)
        else:
            # fallback: как было раньше (по major-тикам)
            for t in ax.get_xticks():
                ax.axvline(t, alpha=0.18, linewidth=0.7, zorder=5)

    # --- profit ---
    if "profit" in df.columns:
        profit = pd.to_numeric(df["profit"], errors="coerce")
        profit_name = "profit"
    elif "profit_total" in df.columns:
        profit = pd.to_numeric(df["profit_total"], errors="coerce")
        profit_name = "profit_total"
    else:
        fut = pd.to_numeric(df.get("profit_fut", 0.0), errors="coerce")
        opt = pd.to_numeric(df.get("profit_opt", 0.0), errors="coerce")
        profit = fut.fillna(0.0) + opt.fillna(0.0)
        profit_name = "profit_fut+profit_opt"

    if profit.notna().sum() < 3:
        return

    # --- signal ---
    if "signal" not in df.columns:
        return
    signal = pd.to_numeric(df["signal"], errors="coerce")

    # --- собрать список feature keys ---
    all_keys = []
    for v in df["features"].tolist():
        if isinstance(v, dict):
            for k in v.keys():
                all_keys.append(str(k))

    if not all_keys:
        return

    # уникальные ключи (стабильно)
    uniq_keys = list(dict.fromkeys(all_keys))

    if max_features is not None:
        uniq_keys = uniq_keys[: int(max_features)]

    # ---------- main loop over feature keys ----------
    for feat_key in uniq_keys:
        # достаём значение features[feat_key] для каждой сделки
        feat_vals = []
        for v in df["features"].tolist():
            if isinstance(v, dict):
                feat_vals.append(_coerce_float(v.get(feat_key)))
            else:
                feat_vals.append(np.nan)

        feat = pd.Series(feat_vals, dtype=float)

        non_nan = int(np.isfinite(feat.to_numpy()).sum())
        if non_nan < 3:
            if debug:
                print(f"[nested_features] skip {feat_key}: non_nan={non_nan}")
            continue

        ncols = 2 if len(signals) > 1 else 1
        nrows = int(math.ceil(len(signals) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(14, 5.8 * nrows),
            dpi=dpi,
            sharey=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1)

        fig.subplots_adjust(top=0.82, hspace=0.55, wspace=0.20)

        mappables = []
        any_plotted = False

        for i, s_val in enumerate(signals):
            ax = axes[i]

            m = (
                feat.notna()
                & profit.notna()
                & signal.notna()
                & (signal.astype(float) == float(s_val))
            )
            cnt = int(m.sum())

            if debug:
                print(f"[nested_features] {feat_key}: signal={s_val} points={cnt}")

            if cnt < 3:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"{feat_key} | signal={s_val}\nnot enough points (n={cnt})",
                    ha="center",
                    va="center",
                )
                continue

            x = feat[m].astype(float).to_numpy()
            y = profit[m].astype(float).to_numpy()

            # если X константа — поддрожим, чтобы hexbin не “исчез”
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            if np.isfinite(x_min) and np.isfinite(x_max) and (x_max <= x_min):
                scale = 1e-6 if abs(x_min) < 1.0 else abs(x_min) * 1e-9
                rng = np.random.default_rng(0)
                x = x + rng.normal(0.0, scale, size=len(x))
                x_min = float(np.nanmin(x))
                x_max = float(np.nanmax(x))

            pear = pd.Series(x).corr(pd.Series(y), method="pearson")
            spear = pd.Series(x).corr(pd.Series(y), method="spearman")

            hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1)
            mappables.append(hb)
            any_plotted = True

            ax.axhline(0.0, linewidth=1.0, zorder=6)
            ax.set_title(
                f"{feat_key} vs {profit_name} | signal={s_val}\n"
                f"Pearson={pear:.4f}  Spearman={spear:.4f}  (n={cnt})"
            )
            ax.set_xlabel(feat_key)
            ax.set_ylabel(profit_name)

            # по умолчанию: нет бинов для вертикальных линий
            bin_edges_for_lines: Optional[np.ndarray] = None

            # --- TOP PANEL: Σprofit per X-bin ---
            if np.isfinite(x_min) and np.isfinite(x_max) and (x_max > x_min):
                edges = np.linspace(x_min, x_max, gridsize + 1)
                centers = (edges[:-1] + edges[1:]) / 2.0
                width = (edges[1] - edges[0]) * 0.9

                # сохраним edges, чтобы вертикальные линии совпали с границами столбиков
                bin_edges_for_lines = edges

                idx = np.digitize(x, edges) - 1
                idx = np.clip(idx, 0, len(centers) - 1)

                sum_profit = np.zeros(len(centers), dtype=float)
                np.add.at(sum_profit, idx, y)

                # counts per bin (n)
                bin_n = np.zeros(len(centers), dtype=int)
                np.add.at(bin_n, idx, 1)

                # верхняя ось: Σprofit (как было)
                ax_top = ax.inset_axes([0.0, 1.11, 1.0, 0.28], transform=ax.transAxes)
                bars = ax_top.bar(centers, sum_profit, width=width, align="center")
                ax_top.axhline(0.0, linewidth=1.0)
                ax_top.set_ylabel("Σ profit", fontsize=9)
                ax_top.tick_params(axis="both", labelsize=8)
                ax_top.set_xticks([])
                ax_top.set_xlim(x_min, x_max)

                y_lo = float(np.nanmin(np.r_[sum_profit, 0.0]))
                y_hi = float(np.nanmax(np.r_[sum_profit, 0.0]))
                y_span = y_hi - y_lo
                pad = (0.10 * y_span) if (y_span > 0) else 1.0
                ax_top.set_ylim(y_lo - pad, y_hi + pad)

                # отдельная "шкала" под столбиками для чисел (n и avg), чтобы ничего не обрезалось
                ax_cnt = ax.inset_axes([0.0, 1.05, 1.0, 0.06], transform=ax.transAxes)
                ax_cnt.set_xlim(x_min, x_max)
                ax_cnt.set_ylim(0.0, 1.0)
                ax_cnt.set_xticks([])
                ax_cnt.set_yticks([])
                for sp in ax_cnt.spines.values():
                    sp.set_visible(False)

                # тонкая линия-разделитель
                ax_cnt.axhline(1.0, color="0.80", linewidth=0.8)

                for k, b in enumerate(bars):
                    n_k = int(bin_n[k])
                    if n_k <= 0:
                        continue

                    avg = float(sum_profit[k]) / float(n_k)
                    cx = b.get_x() + b.get_width() / 2.0

                    # 1) count
                    ax_cnt.text(
                        cx,
                        0.64,
                        str(n_k),
                        ha="center",
                        va="center",
                        fontsize=5,
                        alpha=0.92,
                        clip_on=True,
                    )

                    # 2) avg per trade (under count)
                    ax_cnt.text(
                        cx,
                        0.22,
                        f"({avg:+.0f})",
                        ha="center",
                        va="center",
                        fontsize=5,
                        alpha=0.92,
                        clip_on=True,
                    )

                ax.set_xlim(x_min, x_max)

            # декор (после set_xlim!) + линии по границам бинов
            _dense_x_axis(ax, bin_edges=bin_edges_for_lines)

        # выключить лишние axes
        for j in range(len(signals), len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Nested features: {feat_key}", y=0.98, fontsize=12)

        # резервируем место справа и выносим colorbar в отдельную ось,
        # чтобы шкала НЕ наползала на правый график.
        fig.tight_layout(rect=[0, 0.10, 0.90, 0.98])
        if any_plotted and len(mappables) > 0:
            cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
            fig.colorbar(mappables[-1], cax=cax)

        if save:
            out_dir = save_dir or "."
            os.makedirs(out_dir, exist_ok=True)
            safe_name = "".join(
                ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in feat_key
            )
            fig.savefig(
                os.path.join(out_dir, f"nested_feat_{safe_name}_profit_maps.png"),
                dpi=dpi,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)
      
# def _plot_nested_features_profit_maps(
#     trades: List[Dict[str, Any]],
#     *,
#     show: bool = True,
#     save: bool = False,
#     save_dir: Optional[str] = None,
#     dpi: int = 140,
#     gridsize: int = 45,
#     signals: Tuple[int, int] = (1, 2),
#     max_features: Optional[int] = None,
#     debug: bool = False,
# ) -> None:
#     """
#     Визуализирует hexbin "feature_value vs profit" для ВСЕХ значений из trades[i]['features'].

#     Для каждого feature_name строится figure:
#       - 2 subplot'а: отдельно signal=1 и signal=2 (или signals=...)
#       - X: значение feature
#       - Y: profit
#       - сверху: Σprofit по X-бинам
#       - плотная нижняя шкала X + вертикальные полупрозрачные линии по тикам

#     Параметры:
#       trades: list[dict] - список сделок
#       signals: какие сигналы рисовать (по умолчанию (1,2))
#       max_features: ограничить число фич, если их много (None = все)
#       debug: печатать диагностику
#     """


#     if not trades:
#         return

#     df = pd.DataFrame(trades)
#     if df is None or len(df) == 0:
#         return

#     if "features" not in df.columns:
#         return

#     # ---------- helpers ----------
#     def _coerce_float(v):
#         if v is None:
#             return np.nan
#         try:
#             if isinstance(v, np.generic):
#                 return float(v)
#         except Exception:
#             pass
#         if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
#             v = v[0]
#             try:
#                 if isinstance(v, np.generic):
#                     return float(v)
#             except Exception:
#                 pass
#         try:
#             return float(v)
#         except Exception:
#             return np.nan

#     def _dense_x_axis(ax) -> None:
#         ax.xaxis.set_major_locator(MaxNLocator(nbins=30, min_n_ticks=16))
#         ax.xaxis.set_minor_locator(AutoMinorLocator(2))
#         ax.tick_params(axis="x", which="major", labelsize=5, pad=0, length=3)
#         ax.tick_params(axis="x", which="minor", length=2, width=0.8)

#         for lab in ax.get_xticklabels(which="major"):
#             lab.set_rotation(90)
#             lab.set_horizontalalignment("center")
#             lab.set_verticalalignment("top")
#             lab.set_rotation_mode("anchor")

#         ax.margins(x=0)

#         # вертикальные линии по major-тикам
#         for t in ax.get_xticks():
#             ax.axvline(t, alpha=0.18, linewidth=0.7, zorder=5)

#     # --- profit ---
#     if "profit" in df.columns:
#         profit = pd.to_numeric(df["profit"], errors="coerce")
#         profit_name = "profit"
#     elif "profit_total" in df.columns:
#         profit = pd.to_numeric(df["profit_total"], errors="coerce")
#         profit_name = "profit_total"
#     else:
#         fut = pd.to_numeric(df.get("profit_fut", 0.0), errors="coerce")
#         opt = pd.to_numeric(df.get("profit_opt", 0.0), errors="coerce")
#         profit = fut.fillna(0.0) + opt.fillna(0.0)
#         profit_name = "profit_fut+profit_opt"

#     if profit.notna().sum() < 3:
#         return

#     # --- signal ---
#     if "signal" not in df.columns:
#         return
#     signal = pd.to_numeric(df["signal"], errors="coerce")

#     # --- собрать список feature keys ---
#     all_keys = []
#     for v in df["features"].tolist():
#         if isinstance(v, dict):
#             for k in v.keys():
#                 all_keys.append(str(k))

#     if not all_keys:
#         return

#     # уникальные ключи (стабильно, но без сортировки по алфавиту можно оставить как есть)
#     uniq_keys = list(dict.fromkeys(all_keys))

#     if max_features is not None:
#         uniq_keys = uniq_keys[: int(max_features)]

#     # ---------- main loop over feature keys ----------
#     for feat_key in uniq_keys:
#         # достаём значение features[feat_key] для каждой сделки
#         feat_vals = []
#         for v in df["features"].tolist():
#             if isinstance(v, dict):
#                 feat_vals.append(_coerce_float(v.get(feat_key)))
#             else:
#                 feat_vals.append(np.nan)

#         feat = pd.Series(feat_vals, dtype=float)

#         non_nan = int(np.isfinite(feat.to_numpy()).sum())
#         if non_nan < 3:
#             if debug:
#                 print(f"[nested_features] skip {feat_key}: non_nan={non_nan}")
#             continue

#         ncols = 2 if len(signals) > 1 else 1
#         nrows = int(math.ceil(len(signals) / ncols))

#         fig, axes = plt.subplots(
#             nrows,
#             ncols,
#             figsize=(14, 5.8 * nrows),
#             dpi=dpi,
#             sharey=True,
#         )
#         if not isinstance(axes, np.ndarray):
#             axes = np.array([axes])
#         axes = axes.reshape(-1)

#         fig.subplots_adjust(top=0.82, hspace=0.55, wspace=0.20)

#         mappables = []
#         any_plotted = False

#         for i, s_val in enumerate(signals):
#             ax = axes[i]

#             m = (
#                 feat.notna()
#                 & profit.notna()
#                 & signal.notna()
#                 & (signal.astype(float) == float(s_val))
#             )
#             cnt = int(m.sum())

#             if debug:
#                 print(f"[nested_features] {feat_key}: signal={s_val} points={cnt}")

#             if cnt < 3:
#                 ax.axis("off")
#                 ax.text(
#                     0.5,
#                     0.5,
#                     f"{feat_key} | signal={s_val}\nnot enough points (n={cnt})",
#                     ha="center",
#                     va="center",
#                 )
#                 continue

#             x = feat[m].astype(float).to_numpy()
#             y = profit[m].astype(float).to_numpy()

#             # если X константа — поддрожим, чтобы hexbin не “исчез”
#             x_min = float(np.nanmin(x))
#             x_max = float(np.nanmax(x))
#             if np.isfinite(x_min) and np.isfinite(x_max) and (x_max <= x_min):
#                 scale = 1e-6 if abs(x_min) < 1.0 else abs(x_min) * 1e-9
#                 rng = np.random.default_rng(0)
#                 x = x + rng.normal(0.0, scale, size=len(x))
#                 x_min = float(np.nanmin(x))
#                 x_max = float(np.nanmax(x))

#             pear = pd.Series(x).corr(pd.Series(y), method="pearson")
#             spear = pd.Series(x).corr(pd.Series(y), method="spearman")

#             hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1)
#             mappables.append(hb)
#             any_plotted = True

#             ax.axhline(0.0, linewidth=1.0, zorder=6)
#             ax.set_title(
#                 f"{feat_key} vs {profit_name} | signal={s_val}\n"
#                 f"Pearson={pear:.4f}  Spearman={spear:.4f}  (n={cnt})"
#             )
#             ax.set_xlabel(feat_key)
#             ax.set_ylabel(profit_name)

#             # --- TOP PANEL: Σprofit per X-bin ---
#             if np.isfinite(x_min) and np.isfinite(x_max) and (x_max > x_min):
#                 edges = np.linspace(x_min, x_max, gridsize + 1)
#                 centers = (edges[:-1] + edges[1:]) / 2.0
#                 width = (edges[1] - edges[0]) * 0.9

#                 idx = np.digitize(x, edges) - 1
#                 idx = np.clip(idx, 0, len(centers) - 1)

#                 sum_profit = np.zeros(len(centers), dtype=float)
#                 np.add.at(sum_profit, idx, y)

#                 # counts per bin (n)
#                 bin_n = np.zeros(len(centers), dtype=int)
#                 np.add.at(bin_n, idx, 1)

#                 # верхняя ось: Σprofit (как было)
#                 ax_top = ax.inset_axes([0.0, 1.11, 1.0, 0.28], transform=ax.transAxes)
#                 bars = ax_top.bar(centers, sum_profit, width=width, align="center")
#                 ax_top.axhline(0.0, linewidth=1.0)
#                 ax_top.set_ylabel("Σ profit", fontsize=9)
#                 ax_top.tick_params(axis="both", labelsize=8)
#                 ax_top.set_xticks([])
#                 ax_top.set_xlim(x_min, x_max)

#                 y_lo = float(np.nanmin(np.r_[sum_profit, 0.0]))
#                 y_hi = float(np.nanmax(np.r_[sum_profit, 0.0]))
#                 y_span = y_hi - y_lo
#                 pad = (0.10 * y_span) if (y_span > 0) else 1.0
#                 ax_top.set_ylim(y_lo - pad, y_hi + pad)

#                 # отдельная "шкала" под столбиками для чисел (n и avg), чтобы ничего не обрезалось
#                 ax_cnt = ax.inset_axes([0.0, 1.05, 1.0, 0.06], transform=ax.transAxes)
#                 ax_cnt.set_xlim(x_min, x_max)
#                 ax_cnt.set_ylim(0.0, 1.0)
#                 ax_cnt.set_xticks([])
#                 ax_cnt.set_yticks([])
#                 for sp in ax_cnt.spines.values():
#                     sp.set_visible(False)

#                 # тонкая линия-разделитель
#                 ax_cnt.axhline(1.0, color="0.80", linewidth=0.8)

#                 for k, b in enumerate(bars):
#                     n_k = int(bin_n[k])
#                     if n_k <= 0:
#                         continue

#                     avg = float(sum_profit[k]) / float(n_k)
#                     cx = b.get_x() + b.get_width() / 2.0

#                     # 1) count
#                     ax_cnt.text(
#                         cx,
#                         0.64,
#                         str(n_k),
#                         ha="center",
#                         va="center",
#                         fontsize=5,
#                         alpha=0.92,
#                         clip_on=True,
#                     )

#                     # 2) avg per trade (under count)
#                     ax_cnt.text(
#                         cx,
#                         0.22,
#                         f"({avg:+.0f})",
#                         ha="center",
#                         va="center",
#                         fontsize=5,
#                         alpha=0.92,
#                         clip_on=True,
#                     )

#                 ax.set_xlim(x_min, x_max)

#             # декор (после set_xlim!)
#             _dense_x_axis(ax)

#         # выключить лишние axes
#         for j in range(len(signals), len(axes)):
#             axes[j].axis("off")

#         fig.suptitle(f"Nested features: {feat_key}", y=0.98, fontsize=12)

#         # резервируем место справа и выносим colorbar в отдельную ось,
#         # чтобы шкала НЕ наползала на правый график.
#         fig.tight_layout(rect=[0, 0.10, 0.90, 0.98])
#         if any_plotted and len(mappables) > 0:
#             cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
#             fig.colorbar(mappables[-1], cax=cax)

#         if save:
#             out_dir = save_dir or "."
#             os.makedirs(out_dir, exist_ok=True)
#             safe_name = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in feat_key)
#             fig.savefig(
#                 os.path.join(out_dir, f"nested_feat_{safe_name}_profit_maps.png"),
#                 dpi=dpi,
#                 bbox_inches="tight",
#             )

#         if show:
#             plt.show()
#         else:
#             plt.close(fig)


def _plot_monthday_signal_profit_heatmap(
    trades: List[Dict[str, Any]],
    *,
    show: bool = True,
    save: bool = False,
    save_dir: Optional[str] = None,
    dpi: int = 140,
) -> None:
    """
    Heatmap "месяц x день месяца" на весь период trades.

    Сетка:
      - Y: дни месяца 1..31
      - X: месяцы от min(open_time) до max(open_time) включительно (точно считает)

    В каждой клетке:
      - цвет = sum(profit) (красный<0, белый~0, зелёный>0)
      - цифра внутри = signal (1/2) (только если в клетке есть сделки и это будний день)

    ВАЖНОЕ ИЗМЕНЕНИЕ:
      - все Saturday/Sunday (валидные даты) закрашиваются ЧЁРНЫМ (0,0,0),
        независимо от наличия сделок (считаем, что в выходные не торгуем).
      - надписи Mon/Tue/... УБРАНЫ.

    Пустые дни (нет сделок) -> белая клетка.
    Невалидные даты (например 31 Apr) -> белая клетка без текста.
    """

    if not trades:
        return

    df = pd.DataFrame(trades)
    if df is None or len(df) == 0:
        return

    if "open_time" not in df.columns or "signal" not in df.columns:
        return

    # --- profit ---
    if "profit" in df.columns:
        profit = pd.to_numeric(df["profit"], errors="coerce")
        profit_name = "profit"
    elif "profit_total" in df.columns:
        profit = pd.to_numeric(df["profit_total"], errors="coerce")
        profit_name = "profit_total"
    else:
        fut = pd.to_numeric(df.get("profit_fut", 0.0), errors="coerce")
        opt = pd.to_numeric(df.get("profit_opt", 0.0), errors="coerce")
        profit = fut.fillna(0.0) + opt.fillna(0.0)
        profit_name = "profit_fut+profit_opt"

    signal = pd.to_numeric(df["signal"], errors="coerce")

    # --- datetime from open_time (epoch seconds) in UTC ---
    ot = df["open_time"]
    if pd.api.types.is_numeric_dtype(ot):
        dt = pd.to_datetime(ot.astype(float), unit="s", utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(ot, utc=True, errors="coerce")

    m_valid = dt.notna() & profit.notna() & signal.notna()
    if int(m_valid.sum()) < 3:
        return

    dt = dt[m_valid]
    profit = profit[m_valid].astype(float)
    signal = signal[m_valid].astype(float)

    # --- month index range (точно) ---
    # month_id = year*12 + (month-1)
    month_id = (dt.dt.year.astype(int) * 12 + (dt.dt.month.astype(int) - 1)).to_numpy()
    day = dt.dt.day.astype(int).to_numpy()  # 1..31

    min_mid = int(np.min(month_id))
    max_mid = int(np.max(month_id))
    n_months = (max_mid - min_mid) + 1
    n_days = 31

    # --- accumulate profit sums ---
    sum_total = np.zeros((n_days, n_months), dtype=float)
    cnt_total = np.zeros((n_days, n_months), dtype=int)

    sum_s1 = np.zeros((n_days, n_months), dtype=float)
    cnt_s1 = np.zeros((n_days, n_months), dtype=int)

    sum_s2 = np.zeros((n_days, n_months), dtype=float)
    cnt_s2 = np.zeros((n_days, n_months), dtype=int)

    for mid, d, s, p in zip(month_id, day, signal.to_numpy(), profit.to_numpy()):
        if d < 1 or d > 31:
            continue
        x = int(mid - min_mid)
        y = int(d - 1)

        sum_total[y, x] += float(p)
        cnt_total[y, x] += 1

        si = int(round(float(s)))
        if si == 1:
            sum_s1[y, x] += float(p)
            cnt_s1[y, x] += 1
        elif si == 2:
            sum_s2[y, x] += float(p)
            cnt_s2[y, x] += 1
        else:
            # прочие сигналы не рисуем цифрой
            pass

    has_any = cnt_total > 0

    # --- signal label per cell (only if trades exist) ---
    sig_label = np.full((n_days, n_months), np.nan, dtype=float)
    for y in range(n_days):
        for x in range(n_months):
            if not has_any[y, x]:
                continue
            c1 = cnt_s1[y, x]
            c2 = cnt_s2[y, x]
            if c1 > 0 and c2 == 0:
                sig_label[y, x] = 1
            elif c2 > 0 and c1 == 0:
                sig_label[y, x] = 2
            elif c1 > 0 and c2 > 0:
                # оба сигнала: доминирующий по |sum_profit_signal|
                sig_label[y, x] = 1 if abs(sum_s1[y, x]) >= abs(sum_s2[y, x]) else 2
            else:
                sig_label[y, x] = np.nan

    # --- build validity + weekday grid for each cell ---
    valid_date = np.zeros((n_days, n_months), dtype=bool)
    weekday = np.full((n_days, n_months), -1, dtype=int)  # 0=Mon..6=Sun

    for x in range(n_months):
        mid = min_mid + x
        yy = mid // 12
        mm = (mid % 12) + 1
        for y in range(n_days):
            dd = y + 1
            ts = pd.to_datetime(f"{yy:04d}-{mm:02d}-{dd:02d}", utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            valid_date[y, x] = True
            weekday[y, x] = int(ts.weekday())

    weekend = valid_date & ((weekday == 5) | (weekday == 6))  # Sat/Sun
    weekday_ok = valid_date & (~weekend)

    # --- colour mapping for profit on weekdays only ---
    # matrix for norm computations: consider only weekday cells that have trades
    profit_mat = np.full((n_days, n_months), np.nan, dtype=float)
    profit_mat[has_any] = sum_total[has_any]

    # diverging cmap: red -> white -> green
    cmap = plt.get_cmap("RdYlGn").copy()

    # scale around zero using max |profit|
    mask_for_scale = weekday_ok & has_any
    if np.any(mask_for_scale):
        vmin = float(np.nanmin(profit_mat[mask_for_scale]))
        vmax = float(np.nanmax(profit_mat[mask_for_scale]))
        max_abs = max(abs(vmin), abs(vmax))
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0
    else:
        max_abs = 1.0

    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

    # --- build RGBA image ---
    rgba = np.ones((n_days, n_months, 4), dtype=float)  # default white
    rgba[..., 3] = 1.0

    # invalid dates stay white (already)
    # weekends -> black
    rgba[weekend, 0:3] = 0.75

    # weekdays:
    #   - if no trades -> white
    #   - if trades -> colour by sum profit
    ys, xs = np.where(weekday_ok & has_any)
    for y, x in zip(ys, xs):
        val = profit_mat[y, x]
        r, g, b, a = cmap(norm(val))
        rgba[y, x, 0] = r
        rgba[y, x, 1] = g
        rgba[y, x, 2] = b
        rgba[y, x, 3] = a

    # --- month labels ---
    months = []
    for k in range(n_months):
        mid = min_mid + k
        yy = mid // 12
        mm = (mid % 12) + 1
        months.append(f"{yy:04d}-{mm:02d}")

    # --- plot ---
    fig_w = max(14, n_months * 0.26)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 9.0), dpi=dpi)

    # show RGBA directly (so weekends can be true black)
    ax.imshow(rgba, aspect="auto", origin="lower", interpolation="nearest")

    ax.set_title(f"Month x Day heatmap: colour=sum({profit_name}), weekends=black, digit=signal(1/2)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Day of month")

    ax.set_yticks(np.arange(n_days))
    ax.set_yticklabels([str(i) for i in range(1, n_days + 1)], fontsize=7)

    ax.set_xticks(np.arange(n_months))
    ax.set_xticklabels(months, rotation=90, fontsize=6)

    # grid
    ax.set_xticks(np.arange(-0.5, n_months, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_days, 1), minor=True)
    ax.grid(which="minor", linewidth=0.4, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # --- write signal digit only on weekday cells with trades ---
    for y in range(n_days):
        for x in range(n_months):
            if not (weekday_ok[y, x] and has_any[y, x]):
                continue
            lab = sig_label[y, x]
            if not np.isfinite(lab):
                continue

            # choose text colour based on background luminance
            r, g, b = rgba[y, x, 0], rgba[y, x, 1], rgba[y, x, 2]
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            txt_color = "white" if luminance < 0.45 else "black"

            ax.text(
                x, y, str(int(lab)),
                ha="center", va="center",
                fontsize=6,
                color=txt_color,
                alpha=0.95,
            )

    # --- colourbar for profit scale (does not represent weekend-black) ---
    
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"sum({profit_name}) per (month, day)")

    fig.tight_layout()

    if save:
        out_dir = save_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            os.path.join(out_dir, "month_day_profit_heatmap_weekends_black_signal_digits.png"),
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
        
        
def _pretty_print_stats(stats: Dict[str, Any]) -> None:
    totals = stats["totals"]
    dd = stats["drawdowns"]

    lines: List[str] = []
    lines.append("=== TRADES SUMMARY ===")
    lines.append(f"Total trades         : {stats['n_trades']}")
    lines.append(f"Winrate              : {stats['outcomes']['winrate_pct']:.2f}%")
    lines.append(f"Avg profit / trade   : {_fmt_money(stats['outcomes']['avg_profit_per_trade'])}")
    lines.append("")
    lines.append("--- Totals ---")
    lines.append(f"FUT profit total     : {_fmt_money(totals['profit_fut_total'])}")
    if stats["has_opt"]:
        lines.append(f"OPT profit total     : {_fmt_money(totals['profit_opt_total'])}")
    lines.append(f"TOTAL profit         : {_fmt_money(totals['profit_total'])}")
    lines.append("")
    lines.append("--- Max Drawdowns (absolute) ---")
    lines.append(f"FUT MDD              : {_fmt_money(dd['fut']['mdd'])}")
    if stats["has_opt"] and dd["opt"] is not None:
        lines.append(f"OPT MDD              : {_fmt_money(dd['opt']['mdd'])}")
    lines.append(f"TOTAL MDD            : {_fmt_money(dd['total']['mdd'])}")
    lines.append("")
    lines.append("--- By type_of_close ---")
    for row in stats["by_type_of_close"]:
        lines.append(
            f"{row['type_of_close']}: count={row['count']}, "
            f"fut={_fmt_money(row['profit_fut'])}, "
            f"opt={_fmt_money(row['profit_opt'])}, "
            f"total={_fmt_money(row['profit_total'])}"
        )
    lines.append("")
    lines.append("--- By signal ---")
    for row in stats["by_signal"]:
        lines.append(
            f"{row['signal']}: count={row['count']}, "
            f"fut={_fmt_money(row['profit_fut'])}, "
            f"opt={_fmt_money(row['profit_opt'])}, "
            f"total={_fmt_money(row['profit_total'])}"
        )
    lines.append("")
    lines.append("--- By weekday (Mon..Sun) ---")
    for row in stats["by_weekday"]:
        lines.append(
            f"{row['weekday']}: count={row['count']}, "
            f"winrate={row['winrate_pct']:.1f}%, "
            f"avg={_fmt_money(row['avg_profit_per_trade'])}, "
            f"fut={_fmt_money(row['profit_fut'])}, "
            f"opt={_fmt_money(row['profit_opt'])}, "
            f"total={_fmt_money(row['profit_total'])}"
        )

    # Краткий вывод по index / squeeze_count / atr / rsi / iv
    idx_stats = stats.get("index_stats", {})
    sq_stats = stats.get("squeeze_stats", {})
    atr_stats = stats.get("atr_stats", {})
    rsi_stats = stats.get("rsi_stats", {})
    iv_stats = stats.get("iv_stats", {})

    any_feat = any(
        (d.get("count", 0) or 0) > 0
        for d in [idx_stats, sq_stats, atr_stats, rsi_stats, iv_stats]
    )

    if any_feat:
        lines.append("")
        lines.append("--- Index / Squeeze / ATR / RSI / IV stats ---")

        if idx_stats.get("count", 0) > 0:
            lines.append(
                "index: count={count}, mean={mean:.4f}, median={median:.4f}, "
                "min={min:.4f}, max={max:.4f}, std={std:.4f}, corr_with_profit={corr}".format(
                    count=idx_stats["count"],
                    mean=idx_stats["mean"] if idx_stats["mean"] is not None else float("nan"),
                    median=idx_stats["median"] if idx_stats["median"] is not None else float("nan"),
                    min=idx_stats["min"] if idx_stats["min"] is not None else float("nan"),
                    max=idx_stats["max"] if idx_stats["max"] is not None else float("nan"),
                    std=idx_stats["std"] if idx_stats["std"] is not None else float("nan"),
                    corr=(
                        f"{idx_stats['corr_with_profit_total']:.4f}"
                        if idx_stats["corr_with_profit_total"] is not None
                        else "None"
                    ),
                )
            )

        if sq_stats.get("count", 0) > 0:
            lines.append(
                "squeeze_count: count={count}, mean={mean:.4f}, median={median:.4f}, "
                "min={min:.4f}, max={max:.4f}, std={std:.4f}, corr_with_profit={corr}".format(
                    count=sq_stats["count"],
                    mean=sq_stats["mean"] if sq_stats["mean"] is not None else float("nan"),
                    median=sq_stats["median"] if sq_stats["median"] is not None else float("nan"),
                    min=sq_stats["min"] if sq_stats["min"] is not None else float("nan"),
                    max=sq_stats["max"] if sq_stats["max"] is not None else float("nan"),
                    std=sq_stats["std"] if sq_stats["std"] is not None else float("nan"),
                    corr=(
                        f"{sq_stats['corr_with_profit_total']:.4f}"
                        if sq_stats["corr_with_profit_total"] is not None
                        else "None"
                    ),
                )
            )

        if atr_stats.get("count", 0) > 0:
            lines.append(
                "ATR: count={count}, mean={mean:.4f}, median={median:.4f}, "
                "min={min:.4f}, max={max:.4f}, std={std:.4f}, corr_with_profit={corr}".format(
                    count=atr_stats["count"],
                    mean=atr_stats["mean"] if atr_stats["mean"] is not None else float("nan"),
                    median=atr_stats["median"] if atr_stats["median"] is not None else float("nan"),
                    min=atr_stats["min"] if atr_stats["min"] is not None else float("nan"),
                    max=atr_stats["max"] if atr_stats["max"] is not None else float("nan"),
                    std=atr_stats["std"] if atr_stats["std"] is not None else float("nan"),
                    corr=(
                        f"{atr_stats['corr_with_profit_total']:.4f}"
                        if atr_stats["corr_with_profit_total"] is not None
                        else "None"
                    ),
                )
            )

        if rsi_stats.get("count", 0) > 0:
            lines.append(
                "RSI: count={count}, mean={mean:.4f}, median={median:.4f}, "
                "min={min:.4f}, max={max:.4f}, std={std:.4f}, corr_with_profit={corr}".format(
                    count=rsi_stats["count"],
                    mean=rsi_stats["mean"] if rsi_stats["mean"] is not None else float("nan"),
                    median=rsi_stats["median"] if rsi_stats["median"] is not None else float("nan"),
                    min=rsi_stats["min"] if rsi_stats["min"] is not None else float("nan"),
                    max=rsi_stats["max"] if rsi_stats["max"] is not None else float("nan"),
                    std=rsi_stats["std"] if rsi_stats["std"] is not None else float("nan"),
                    corr=(
                        f"{rsi_stats['corr_with_profit_total']:.4f}"
                        if rsi_stats["corr_with_profit_total"] is not None
                        else "None"
                    ),
                )
            )

        if iv_stats.get("count", 0) > 0:
            lines.append(
                "IV: count={count}, mean={mean:.4f}, median={median:.4f}, "
                "min={min:.4f}, max={max:.4f}, std={std:.4f}, corr_with_profit={corr}".format(
                    count=iv_stats["count"],
                    mean=iv_stats["mean"] if iv_stats["mean"] is not None else float("nan"),
                    median=iv_stats["median"] if iv_stats["median"] is not None else float("nan"),
                    min=iv_stats["min"] if iv_stats["min"] is not None else float("nan"),
                    max=iv_stats["max"] if iv_stats["max"] is not None else float("nan"),
                    std=iv_stats["std"] if iv_stats["std"] is not None else float("nan"),
                    corr=(
                        f"{iv_stats['corr_with_profit_total']:.4f}"
                        if iv_stats["corr_with_profit_total"] is not None
                        else "None"
                    ),
                )
            )

    print("\n".join(lines))
# =============================================================================
# Main entry: analyse_trades
# =============================================================================

def analyse_trades(
    trades: List[Dict[str, Any]],
    show: bool = True,
    save: bool = False,
    save_dir: Optional[str] = None,
    print_stats: bool = False,
    title: Optional[str] = None,
    dpi: int = 140,
    *,
    open_browser: bool = True,
    overwrite_png: bool = True,
    serve: bool = True,
    dotenv_path: str = ENV_PATH_DEFAULT,
    block_server: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    В режиме save=True:
      - сохраняет ВСЕ графики в один каталог (перезаписывает PNG)
      - пишет images.js (для статичного index.html)
      - опционально поднимает локальный сервер и открывает браузер
      - поддерживает кнопку "Send to Telegram" через /api/send

    КРИТИЧНО:
      Если запускаешь как обычный скрипт, процесс Python должен жить,
      иначе сервер умрёт и браузер увидит "site can’t be reached".
      Для этого есть block_server (по умолчанию авто).
    """
    import os
    import glob
    import webbrowser

    stats, df = compute_trade_stats(trades)

    if save_dir is None:
        save_dir = REPORT_DIR_DEFAULT

    if save:
        os.makedirs(save_dir, exist_ok=True)

        if overwrite_png:
            for fp in glob.glob(os.path.join(save_dir, "*.png")):
                try:
                    os.remove(fp)
                except Exception:
                    pass
            js_fp = os.path.join(save_dir, "images.js")
            try:
                if os.path.exists(js_fp):
                    os.remove(js_fp)
            except Exception:
                pass

        tmp_dir = os.path.join(save_dir, "_tmp")
        all_pngs: List[str] = []

        all_pngs += _run_plot_save_to_tmp_and_move(
            plot_fn=lambda td: _plot_equity(df, stats, title=title, show=False, save=True, save_dir=td, dpi=dpi),
            tmp_dir=tmp_dir,
            out_dir=save_dir,
            base_name="01_equity",
        )

        all_pngs += _run_plot_save_to_tmp_and_move(
            plot_fn=lambda td: _plot_distributions(df, stats, show=False, save=True, save_dir=td, dpi=dpi),
            tmp_dir=tmp_dir,
            out_dir=save_dir,
            base_name="02_distributions",
        )

        all_pngs += _run_plot_save_to_tmp_and_move(
            plot_fn=lambda td: _plot_weekday_stats(df, stats, show=False, save=True, save_dir=td, dpi=dpi),
            tmp_dir=tmp_dir,
            out_dir=save_dir,
            base_name="03_weekday_stats",
        )

        all_pngs += _run_plot_save_to_tmp_and_move(
            plot_fn=lambda td: _plot_trades_timeline(df, stats, show=False, save=True, save_dir=td, dpi=dpi),
            tmp_dir=tmp_dir,
            out_dir=save_dir,
            base_name="04_trades_timeline",
        )

        all_pngs += _run_plot_save_to_tmp_and_move(
            plot_fn=lambda td: _plot_nested_features_profit_maps(trades, show=False, save=True, save_dir=td, dpi=dpi),
            tmp_dir=tmp_dir,
            out_dir=save_dir,
            base_name="05_nested_features_profit_maps",
        )

        all_pngs += _run_plot_save_to_tmp_and_move(
            plot_fn=lambda td: _plot_monthday_signal_profit_heatmap(trades, show=False, save=True, save_dir=td, dpi=dpi),
            tmp_dir=tmp_dir,
            out_dir=save_dir,
            base_name="06_monthday_signal_profit_heatmap",
        )

        # прибираем tmp
        try:
            for fp in glob.glob(os.path.join(tmp_dir, "*.png")):
                try:
                    os.remove(fp)
                except Exception:
                    pass
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass
        except Exception:
            pass

        all_pngs_sorted = sorted(all_pngs, key=lambda x: os.path.basename(x))
        images_js_path = _write_images_js(save_dir, all_pngs_sorted)

        stats["report_dir"] = save_dir
        stats["report_index_html"] = os.path.join(save_dir, "index.html")
        stats["report_images_js"] = images_js_path
        stats["report_images"] = all_pngs_sorted

        if open_browser:
            if serve:
                _load_dotenv_simple(dotenv_path)
                import os as _os
                token = _os.getenv("ALGO_BOT")
                chat_id = _os.getenv("CHAT_ID")

                base_url, stop_event = _start_report_server(
                    report_dir=save_dir,
                    telegram_token=token,
                    telegram_chat_id=chat_id,
                )

                ts = int(_time.time() * 1000)
                url = base_url + f"/index.html?v={ts}"
                webbrowser.open_new_tab(url)
                stats["report_url"] = url

                # авто-блокировка: в обычном скрипте блокируем, в ноутбуке — нет
                if block_server is None:
                    block_server = (not _is_interactive_session())

                if block_server:
                    # ждём, пока в браузере нажмут "Stop viewer"
                    stop_event.wait()

            else:
                webbrowser.open_new_tab("file://" + os.path.abspath(os.path.join(save_dir, "index.html")))

    else:
        _plot_equity(df, stats, title=title, show=show, save=False, save_dir=None, dpi=dpi)
        _plot_distributions(df, stats, show=show, save=False, save_dir=None, dpi=dpi)
        _plot_weekday_stats(stats, show=show, save=False, save_dir=None, dpi=dpi)
        _plot_trades_timeline(df, stats, show=show, save=False, save_dir=None, dpi=dpi)
        _plot_nested_features_profit_maps(trades, show=show, save=False, save_dir=None, dpi=dpi)
        _plot_monthday_signal_profit_heatmap(trades, show=show, save=False, save_dir=None, dpi=dpi)

    if print_stats:
        _pretty_print_stats(stats)

    return stats




