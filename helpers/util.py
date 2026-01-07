from __future__ import annotations


from bisect import bisect_right
from typing import Mapping, TypeVar, Optional
import bisect
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from typing import Iterable, Optional, Literal, Dict, Any, List
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import re

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
import csv
import os
from typing import Dict, Any

def append_dict_to_csv(row: Dict[str, Any], csv_path: str) -> None:
    """
    Дописать словарь как строку в CSV-файл.
    - Если файла нет или он пустой: создать, записать header (ключи словаря) и строку.
    - Если файл есть: считать существующий header и дописать строку,
      сопоставляя значения по именам столбцов.

    Поведение при несовпадении ключей:
    - Ключи, которых нет в существующем header, игнорируются.
    - Столбцы из header, которых нет в словаре, заполняются пустой строкой.

    :param row: Данные для записи (ключи — имена столбцов).
    :param csv_path: Путь к CSV-файлу.
    """
    if not isinstance(row, dict):
        raise TypeError("row должен быть dict[str, Any]")

    # Создать директорию при необходимости
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    file_exists = os.path.exists(csv_path)
    file_nonempty = file_exists and os.path.getsize(csv_path) > 0

    if not file_nonempty:
        # Новый файл или пустой: пишем header из ключей словаря и первую строку
        fieldnames = list(row.keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Преобразуем None -> '' чтобы не получить "None" в CSV
            clean_row = {k: ("" if row[k] is None else row[k]) for k in fieldnames}
            writer.writerow(clean_row)
        return

    # Файл существует и не пуст: читаем header
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

    # На случай, если файл существовал, но без корректного header (крайний случай)
    if not header:
        fieldnames = list(row.keys())
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            clean_row = {k: ("" if row[k] is None else row[k]) for k in fieldnames}
            writer.writerow(clean_row)
        return

    # Сопоставляем значения по существующему header, отсутствующие — ''
    out_row = {col: ("" if row.get(col) is None else row.get(col, "")) for col in header}

    # Дописываем строку
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(out_row)

def find_candle_index(timestamp, candles):
    start = 0
    end = len(candles) - 1
    
    while start <= end:
        mid = (start + end) // 2
        if candles[mid][0] == timestamp:
            return mid
        elif candles[mid][0] < timestamp:
            start = mid + 1
        else:
            end = mid - 1
    
    return -1





def load_wf_step_bundles(
    step_dir: str,
    *,
    loader: Callable[[str], Any],
    expected_folds: Optional[int] = 10,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Загружает все fold_XXX директории внутри step_dir и возвращает словарь:
      { "fold_000": bundle, "fold_001": bundle, ... }

    Параметры:
      - step_dir: путь вида "_models/.../step_10"
      - loader: функция загрузки бандла (например load_wf_lgbm_bundle)
      - expected_folds: сколько фолдов ожидаем (по умолчанию 10). None = не проверять.
      - strict: если True — при несовпадении expected_folds кидает ошибку, иначе предупреждает.
    """
    p = Path(step_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"step_dir not found or not a directory: {step_dir}")

    found: list[tuple[int, Path]] = []
    for d in p.iterdir():
        if not d.is_dir():
            continue
        m = re.fullmatch(r"fold_(\d+)", d.name)
        if m:
            found.append((int(m.group(1)), d))

    found.sort(key=lambda x: x[0])

    if expected_folds is not None and len(found) != expected_folds:
        msg = f"Expected {expected_folds} folds in '{step_dir}', but found {len(found)}: {[d.name for _, d in found]}"
        if strict:
            raise ValueError(msg)
        else:
            print(f"[WARN] {msg}")

    bundles: Dict[str, Any] = {}
    for idx, fold_path in found:
        key = f"fold_{idx:03d}"
        bundles[key] = loader(str(fold_path))

    return bundles

def find_bundle_key_by_val_period(
    bundles: Dict[str, Any],
    timestamp_ms: int,
) -> Tuple[Optional[str], Optional[Any]]:
    """
    Возвращает (fold_key, bundle) по попаданию timestamp_ms в val_period.
    Если не найдено: (None, None).
    """
    try:
        ts = int(timestamp_ms)
    except Exception:
        return None, None

    best_key: Optional[str] = None
    best_bundle: Optional[Any] = None
    best_start: Optional[int] = None

    for key in sorted(bundles.keys()):
        b = bundles.get(key)
        if b is None:
            continue

        vp = getattr(b, "val_period", None)
        if not isinstance(vp, dict):
            continue

        try:
            start_ts = int(vp.get("start_ts"))
            end_ts = int(vp.get("end_ts"))
        except Exception:
            continue

        if start_ts <= ts <= end_ts:
            if best_start is None or start_ts > best_start:
                best_start = start_ts
                best_key = key
                best_bundle = b

    return best_key, best_bundle


def load_preds_pandas(file_path):
    """
    Загружает CSV с помощью pandas и возвращает словарь {timestamp_ms: pred_1}.
    Типы данных будут определены автоматически.
    """
    try:
        df = pd.read_csv(file_path)
        
        # Проверка наличия нужных колонок
        if 'timestamp_ms' not in df.columns or 'pred_1' not in df.columns:
            raise ValueError("В файле отсутствуют нужные колонки")

        # Создаем словарь: zip объединяет колонки в пары (ключ, значение)
        return dict(zip(df['timestamp_ms'], df['pred_1']))
        
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        return {}
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return {}
    
    
def calculate_percent_difference(close, high_or_low):
    return (high_or_low - close) / close




@dataclass(frozen=True)
class FundingIndex:
    """
    Удобная обёртка:
      - rates: {calc_time_ms: funding_rate_str}
      - keys_sorted: отсортированные calc_time_ms для быстрого поиска предыдущего
    """
    rates: Dict[int, str]
    keys_sorted: List[int]


def load_funding_csv_to_dict(csv_path: str) -> FundingIndex:
    """
    Загружает BTCUSDT_funding_all.csv в FundingIndex:
      rates[calc_time_ms] = last_funding_rate (как строка, без float преобразований)

    Ожидаемые колонки:
      calc_time,funding_interval_hours,last_funding_rate
    """
    rates: Dict[int, str] = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or no header: {csv_path}")

        # минимальная проверка
        need = {"calc_time", "last_funding_rate"}
        if not need.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV schema mismatch. Need columns {sorted(need)}. Got {reader.fieldnames}")

        for row in reader:
            ct_raw = (row.get("calc_time") or "").strip()
            fr_raw = (row.get("last_funding_rate") or "").strip()
            if not ct_raw or not fr_raw:
                continue

            try:
                ct_ms = int(ct_raw)
            except ValueError:
                continue

            # funding_rate оставляем строкой (как в вашем файле)
            rates[ct_ms] = fr_raw

    if not rates:
        raise ValueError(f"No valid rows loaded from {csv_path}")

    keys_sorted = sorted(rates.keys())
    return FundingIndex(rates=rates, keys_sorted=keys_sorted)


def get_last_funding_rate_at_or_before(
    idx: FundingIndex,
    ts_ms: int,
) -> str:
    """
    Возвращает last_funding_rate:
      - если ts_ms точно есть в idx.rates -> берём его
      - иначе берём последний прошлый по времени (<= ts_ms)

    Пример:
      ts=06:00 -> вернёт funding в 00:00
      ts=10:00 -> вернёт funding в 08:00

    Исключение:
      если ts_ms раньше самого первого calc_time в файле.
    """
    if not isinstance(ts_ms, int):
        try:
            ts_ms = int(ts_ms)
        except Exception as e:
            raise TypeError("ts_ms must be int (ms)") from e

    # точное попадание
    v = idx.rates.get(ts_ms)
    if v is not None:
        return v

    # позиция вставки справа, чтобы получить <=
    pos = bisect.bisect_right(idx.keys_sorted, ts_ms) - 1
    if pos < 0:
        raise KeyError(
            f"Timestamp {ts_ms} is earlier than first available fundingTime {idx.keys_sorted[0]}"
        )

    prev_key = idx.keys_sorted[pos]
    return idx.rates[prev_key]






SmoothMode = Literal["none", "sma", "ewma"]


@dataclass(frozen=True)
class CurveDynamics:
    # Основные знаковые метрики (плюс = вверх, минус = вниз)
    trend: float                 # нормализованная скорость (вверх/вниз)
    bend: float                  # нормализованная кривизна (загиб вверх/вниз)

    # Метрики в "сырых" единицах (если шаг по времени dt=1)
    slope_last: float            # последняя скорость (dy/dt)
    slope_mean: float            # средняя скорость по окну
    accel_last: float            # последнее ускорение (d2y/dt2)
    accel_mean: float            # среднее ускорение по окну

    # Кривизна по квадратичной аппроксимации y = a*x^2 + b*x + c
    quad_a: float                # коэффициент a
    curvature: float             # 2a (d2y/dt2 в модели)

    # Человекочитаемые ярлыки
    trend_label: str             # "up" / "down" / "flat"
    bend_label: str              # "bending_up" / "bending_down" / "flat"

    # Насколько уверенно (0..1) — грубая оценка SNR по окну
    confidence: float

    # Дополнительно
    window_used: int
    smooth_mode: str


def _smooth(y: np.ndarray, mode: SmoothMode, span: int) -> np.ndarray:
    if mode == "none":
        return y

    if span <= 1:
        return y

    if mode == "sma":
        # простая скользящая средняя
        k = span
        kernel = np.ones(k, dtype=np.float64) / k
        # pad, чтобы длина сохранилась
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        ypad = np.pad(y, (pad_left, pad_right), mode="edge")
        return np.convolve(ypad, kernel, mode="valid")

    if mode == "ewma":
        # экспоненциальное сглаживание
        alpha = 2.0 / (span + 1.0)
        out = np.empty_like(y, dtype=np.float64)
        out[0] = float(y[0])
        for i in range(1, len(y)):
            out[i] = alpha * float(y[i]) + (1.0 - alpha) * out[i - 1]
        return out

    raise ValueError(f"Unknown smooth mode: {mode}")


def compute_curve_dynamics(
    values: Iterable[float],
    *,
    window: int = 21,
    dt: float = 1.0,
    smooth: SmoothMode = "ewma",
    smooth_span: int = 5,
    flat_eps: float = 1e-9,
) -> CurveDynamics:
    """
    Оценивает:
      - динамику вверх/вниз (trend): нормализованная скорость по окну (плюс/минус)
      - загиб вверх/вниз (bend): нормализованная кривизна по окну (плюс/минус)

    Интерпретация:
      trend > 0  -> серия в среднем растёт (вверх)
      trend < 0  -> серия в среднем падает (вниз)

      bend > 0   -> кривая "загибается вверх" (ускоряется вверх / становится более выпуклой вверх)
      bend < 0   -> кривая "загибается вниз" (ускоряется вниз / становится более выпуклой вниз)

    window: сколько последних точек использовать (если данных меньше — берём всё).
    dt: шаг по времени между точками (если 1 точка = 1 час, оставляйте dt=1).
    """
    y0 = np.asarray(list(values), dtype=np.float64)
    if y0.size < 3:
        raise ValueError("Need at least 3 points to estimate curve dynamics.")

    w = int(window)
    if w < 3:
        w = 3
    if w > y0.size:
        w = y0.size

    y = y0[-w:]
    y_s = _smooth(y, smooth, smooth_span)

    # 1-я производная (скорость)
    v = np.diff(y_s) / dt  # длина w-1
    slope_last = float(v[-1])
    slope_mean = float(np.mean(v))

    # 2-я производная (ускорение)
    a = np.diff(v) / dt    # длина w-2
    accel_last = float(a[-1])
    accel_mean = float(np.mean(a))

    # Квадратичная аппроксимация по окну: y = a*x^2 + b*x + c
    # x в секундах/часах не важен, важно dt, поэтому используем x = 0..w-1 * dt
    x = (np.arange(w, dtype=np.float64) * dt)
    p = np.polyfit(x, y_s, deg=2)  # p[0]=a, p[1]=b, p[2]=c
    quad_a = float(p[0])
    curvature = float(2.0 * p[0])  # d2y/dt2 для квадратичной модели

    # Нормализация: делим на масштаб данных, чтобы сравнивать разные серии
    scale = float(np.std(y_s))
    if not np.isfinite(scale) or scale < flat_eps:
        scale = 1.0

    trend = slope_mean / scale              # нормализованная "скорость"
    bend = curvature / scale                # нормализованная "кривизна"

    def _label(val: float) -> str:
        if abs(val) <= 1e-12:
            return "flat"
        return "up" if val > 0 else "down"

    trend_label = _label(trend)
    bend_label = "flat"
    if abs(bend) > 1e-12:
        bend_label = "bending_up" if bend > 0 else "bending_down"

    # Простейшая уверенность: отношение дисперсии сглаженного сигнала к шуму остатков
    resid = y - y_s
    sig = float(np.std(y_s))
    noise = float(np.std(resid)) + 1e-12
    snr = sig / noise
    confidence = float(np.clip(snr / (snr + 1.0), 0.0, 1.0))

    return CurveDynamics(
        trend=float(trend),
        bend=float(bend),
        slope_last=slope_last,
        slope_mean=slope_mean,
        accel_last=accel_last,
        accel_mean=accel_mean,
        quad_a=quad_a,
        curvature=curvature,
        trend_label=trend_label,
        bend_label=bend_label,
        confidence=confidence,
        window_used=w,
        smooth_mode=smooth,
    )
def load_csv_as_dict(
    csv_path: str,
    key_col: str,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
    strict_unique: bool = False,
    drop_key_from_row: bool = True,
    drop_constant_and_all_nan: bool = False,
    drop_non_numeric: bool = False,
) -> Dict[Any, Dict[str, Any]]:
    """
    Загружает CSV в словарь:
      { <key_col value>: {col_name: value, ...}, ... }

    Опции:
      - drop_key_from_row: удалить key_col из внутреннего словаря каждой строки (по умолчанию True)
      - drop_constant_and_all_nan:
            удалить колонки (кроме key_col), где:
              (1) все значения пустые/NaN
              (2) все НЕ-пустые значения одинаковые (константа)
      - drop_non_numeric:
            удалить колонки (кроме key_col), где среди НЕ-пустых значений есть НЕ (int/float),
            или вообще нет ни одного числового значения.
            (bool не считается числом для этой проверки)
    """
    import csv
    import math

    _UNSET = object()

    def is_missing(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and math.isnan(v):
            return True
        return False

    def is_number(v: Any) -> bool:
        # bool — подкласс int, но по требованию исключаем
        if isinstance(v, bool):
            return False
        return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))

    def auto_cast(v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return None

        # int
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                pass

        # float
        try:
            x = float(s)
            if not math.isfinite(x) or math.isnan(x):
                return None
            return x
        except ValueError:
            return s

    def cast_key(v: Any) -> Any:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return None

        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                return s

        # например "1765684800000.0"
        try:
            x = float(s)
            if not math.isfinite(x) or math.isnan(x):
                return s
            xr = round(x)
            if abs(x - xr) < 1e-9:
                return int(xr)
            return x
        except ValueError:
            return s

    out: Dict[Any, Dict[str, Any]] = {}

    # Статистика по колонкам для фильтраций (кроме key_col)
    col_has_non_missing: Dict[str, bool] = {}
    col_first_non_missing: Dict[str, Any] = {}
    col_has_different: Dict[str, bool] = {}

    col_has_numeric: Dict[str, bool] = {}
    col_has_non_numeric: Dict[str, bool] = {}

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("CSV пустой или нет строки заголовков (header).")

        fieldnames = list(reader.fieldnames)
        if key_col not in fieldnames:
            raise ValueError(f"В CSV нет колонки {key_col!r}. Колонки: {fieldnames}")

        need_stats = drop_constant_and_all_nan or drop_non_numeric
        if need_stats:
            for c in fieldnames:
                if c == key_col:
                    continue
                col_has_non_missing[c] = False
                col_first_non_missing[c] = _UNSET
                col_has_different[c] = False

                col_has_numeric[c] = False
                col_has_non_numeric[c] = False

        for row in reader:
            raw_key = row.get(key_col)
            key = cast_key(raw_key)
            if key is None:
                raise ValueError(f"Пустой/некорректный ключ в колонке {key_col!r}: {raw_key!r}")

            if strict_unique and key in out:
                raise ValueError(f"Дубликат ключа {key!r} в колонке {key_col!r}")

            row_cast: Dict[str, Any] = {}

            for c in fieldnames:
                if c == key_col and drop_key_from_row:
                    continue

                v = auto_cast(row.get(c))
                row_cast[c] = v

                if need_stats and c != key_col and not is_missing(v):
                    # для constant/all_nan
                    if not col_has_non_missing[c]:
                        col_has_non_missing[c] = True
                        col_first_non_missing[c] = v
                    else:
                        if not col_has_different[c] and v != col_first_non_missing[c]:
                            col_has_different[c] = True

                    # для non-numeric
                    if is_number(v):
                        col_has_numeric[c] = True
                    else:
                        col_has_non_numeric[c] = True

            out[key] = row_cast

    # Формируем список колонок на удаление
    if out and (drop_constant_and_all_nan or drop_non_numeric):
        drop_cols = set()

        for c in fieldnames:
            if c == key_col:
                continue

            if drop_constant_and_all_nan:
                # all NaN
                if not col_has_non_missing.get(c, False):
                    drop_cols.add(c)
                    continue
                # constant (все НЕ-пустые одинаковые)
                if not col_has_different.get(c, False):
                    drop_cols.add(c)
                    continue

            if drop_non_numeric:
                # если среди непустых есть нечисловые ИЛИ вообще нет числовых
                if col_has_non_numeric.get(c, False) or not col_has_numeric.get(c, False):
                    drop_cols.add(c)

        if drop_cols:
            for _, r in out.items():
                for c in drop_cols:
                    r.pop(c, None)

    return out



import numbers

MS_PER_DAY = 86_400_000  # 24 часа в миллисекундах (UTC сутки)


def day_start_ts_ms_utc(timestamp_ms: int) -> int:
    """
    Берёт timestamp в миллисекундах (Unix epoch, UTC) и возвращает timestamp
    начала этих суток (00:00:00.000 UTC) тоже в миллисекундах.

    Работает чисто целочисленно: “отбрасываем” внутри дня часы/минуты/секунды.
    """
    if not isinstance(timestamp_ms, numbers.Integral):
        raise TypeError(f"timestamp_ms must be an integer (ms since epoch), got: {type(timestamp_ms).__name__}")

    return (int(timestamp_ms) // MS_PER_DAY) * MS_PER_DAY


# Пример:
# ts_any = 1735601234567
# print(day_start_ts_ms_utc(ts_any))
# --- colour printing helpers (ANSI terminal) ---

RESET = "\033[0m"
RED   = "\033[31m"
GREEN = "\033[32m"
WHITE = "\033[97m"   # bright white (better on dark background)
YELLOW = "\033[33m"  # fallback for unexpected values


def print_regimes_colored(
    reg_d: int,
    state_like_csv: int,
    reg_h: int,
    *,
    sep: str = " ",
    end: str = "\n",
    use_color: bool = True,
) -> None:
    """
    Печатает три значения в одной строке, раскрашивая по правилам:

    Для первых двух (reg_d, state_like_csv):
      0 -> красный, 1 -> белый, 2 -> зелёный

    Для третьего (reg_h):
      0,1 -> красный; 2,3 -> белый; 4,5 -> зелёный

    Параметры:
      sep      - разделитель между числами (по умолчанию пробел)
      end      - чем заканчивать печать (по умолчанию перенос строки)
      use_color- выключить цвета (например, при логировании в файл)
    """

    def _color_first_two(v: int) -> str:
        if v == 0:
            return RED
        if v == 1:
            return WHITE
        if v == 2:
            return GREEN
        return YELLOW

    def _color_third(v: int) -> str:
        if v in (0, 1):
            return RED
        if v in (2, 3):
            return WHITE
        if v in (4, 5):
            return GREEN
        return YELLOW

    if not use_color:
        print(reg_d, state_like_csv, reg_h, sep=sep, end=end)
        return

    s1 = f"{_color_first_two(reg_d)}{reg_d}{RESET}"
    s2 = f"{_color_first_two(state_like_csv)}{state_like_csv}{RESET}"
    s3 = f"{_color_third(reg_h)}{reg_h}{RESET}"

    print(f"{s1}{sep}{s2}{sep}{s3}", end=end)




T = TypeVar("T")


def get_at_or_before(ts_map: Mapping[int, T], ts_ms: int, default: Optional[T] = None, *, rebuild: bool = False) -> Optional[T]:
    """
    Вернёт значение по точному ts_ms, а если такого ключа нет — значение по ближайшему
    ключу строго ДО ts_ms (последнее доступное "на тот момент").

    Оптимизация для цикла:
    - внутри держит кеш: один раз сортирует ключи и хранит параллельный список значений
    - поиск дальше делается через bisect (очень быстро)

    Важно:
    - если словарь только ДОБАВЛЯЕТСЯ (append-only), кеш сам пересоберётся при изменении len(ts_map)
    - если вы ПЕРЕЗАПИСЫВАЕТЕ существующие значения без изменения длины — передайте rebuild=True один раз
    """
    cache = getattr(get_at_or_before, "_cache", None)
    if cache is None:
        cache = get_at_or_before._cache = {}

    # чтобы кеш не рос бесконечно, если вдруг будут разные словари
    if len(cache) > 8:
        cache.clear()

    d_id = id(ts_map)
    entry = cache.get(d_id)

    if rebuild or entry is None or entry["n"] != len(ts_map):
        items = sorted(((int(k), v) for k, v in ts_map.items()), key=lambda kv: kv[0])
        keys = [k for k, _ in items]
        vals = [v for _, v in items]
        entry = {"n": len(ts_map), "keys": keys, "vals": vals}
        cache[d_id] = entry

    keys = entry["keys"]
    vals = entry["vals"]

    i = bisect_right(keys, int(ts_ms)) - 1
    if i < 0:
        return default
    return vals[i]


def find_index(timestamp, data):

    for i in range(len(data)):
        if data[i][0] > timestamp:
            if i > 0:
                return i - 1
            else:
                print(datetime.fromtimestamp(int(timestamp/1000)))
                return None

    return len(data) - 1


def get_time_segment(start_timestamp, end_timestamp, data):
    result = []
    for sublist in data:
        timestamp = sublist[0]
        if start_timestamp <= timestamp < end_timestamp:
            result.append(sublist)
    return np.array(result)

def combine_last_candle(start_timestamp, end_timestamp, dt):
    data = get_time_segment(start_timestamp, end_timestamp, dt)
    if len(data) < 2:
        return None

    close = data[-1][4]
    highs = data[:, 2]
    lows = data[:, 3]
    open = data[0][1]

    high = np.max(highs)
    low = np.min(lows)

    return [data[0][0], open, high, low, close, 999]