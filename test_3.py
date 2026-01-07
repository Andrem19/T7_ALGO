#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Фильтрует строки CSV по времени (несколько часов) на основе timestamp в миллисекундах
и сохраняет результат в соседний файл.

По умолчанию:
- вход:  ch_res_2.csv
- выход: ch_res_3.csv
- часы:  (6, 12)
- таймзона: UTC  (если нужно Europe/London — поменяйте константу TZ_NAME)
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore


# ===================== НАСТРОЙКИ =====================
INPUT_PATH = Path("vector.csv")
OUTPUT_PATH = Path("vector_6.csv")

TARGET_HOURS = (6,)  # можно (6,) или (6, 12, 18) и т.д.
TZ_NAME = "UTC"  # например: "Europe/London"

# Имя колонки с timestamp в миллисекундах. Скрипт сам найдёт первую из этих, которая есть в файле.
TS_COL_CANDIDATES = ("tm_ts", "tm_ms")
# =====================================================


def _parse_ms(value: str) -> int:
    """
    Превращает строку из CSV в целое число миллисекунд.
    Поддерживает значения вида "1616547600000.0".
    """
    s = (value or "").strip()
    if not s:
        raise ValueError("Empty timestamp value")
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def _get_tz(tz_name: str):
    if tz_name.upper() == "UTC":
        return timezone.utc
    if ZoneInfo is None:
        raise RuntimeError(
            "zoneinfo недоступен в этой версии Python. "
            "Используйте TZ_NAME='UTC' или обновите Python до 3.9+."
        )
    return ZoneInfo(tz_name)


def filter_rows_at_hours(
    input_path: Path,
    output_path: Path,
    *,
    target_hours: tuple[int, ...] = (6,),
    tz_name: str = "UTC",
    ts_col_candidates: tuple[str, ...] = ("tm_ts", "tm_ms"),
) -> tuple[int, int]:
    """
    Читает CSV, оставляет строки, где timestamp (ms) попадает на один из target_hours
    в указанной таймзоне, и пишет новый CSV.

    Возвращает (total_rows, kept_rows) без учёта хедера.
    """
    if not target_hours:
        raise ValueError("target_hours must be non-empty, e.g. (6,) or (6, 12).")

    # нормализация + быстрая проверка принадлежности
    hours_set = set()
    for h in target_hours:
        if not isinstance(h, int):
            raise TypeError(f"Each hour must be int, got: {type(h)}")
        if h < 0 or h > 23:
            raise ValueError(f"Hour must be in 0..23, got: {h}")
        hours_set.add(h)

    tz = _get_tz(tz_name)

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header (fieldnames).")

        ts_col = next((c for c in ts_col_candidates if c in reader.fieldnames), None)
        if ts_col is None:
            raise ValueError(
                f"Не нашёл колонку с timestamp. Ожидал одну из: {ts_col_candidates}. "
                f"В файле есть: {tuple(reader.fieldnames)}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            total = 0
            kept = 0

            for row in reader:
                total += 1

                ms = _parse_ms(row[ts_col])
                dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(tz)

                if dt.hour in hours_set:
                    writer.writerow(row)
                    kept += 1

    return total, kept


if __name__ == "__main__":
    total_rows, kept_rows = filter_rows_at_hours(
        INPUT_PATH,
        OUTPUT_PATH,
        target_hours=TARGET_HOURS,
        tz_name=TZ_NAME,
        ts_col_candidates=TS_COL_CANDIDATES,
    )
    print(f"Done. Total rows: {total_rows}, kept (hours={TARGET_HOURS}, tz={TZ_NAME}): {kept_rows}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")
