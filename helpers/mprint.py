# mprint.py
# -*- coding: utf-8 -*-
"""
Мини-логгер для цветного вывода в терминал и (опционально) записи в файл.

СОВМЕСТИМОСТЬ СО СТАРОЙ ВЕРСИЕЙ СОХРАНЕНА:
- Экземпляр logger (from mprint import logger) и вызовы logger.red(...), logger.green(...), ...
- Методы по цветам (red/green/white/blue/yellow/magenta/cyan/grey/bright_red/bright_green).
- Параметры b=True (жирный) и time=True (метка времени, локальная).
- Поведение "как print": *args, sep=, end=, stream=, flush=.
- Запись в файл с фильтром цветов: logger.save(save_path, colors=[...]).
  Если colors=None или пусто — сохраняются все цвета. В файл ANSI-коды не пишутся.

НОВОЕ:
- Можно импортировать цветные функции напрямую, без logger:
    from mprint import red, green, blue
    red("Hello", time=True)
    blue("part1", nl=False)
    green("part2")

- Также доступны прямые обёртки над сервисными методами:
    from mprint import save, disable_save, set_time_format, set_default_time, force_color, log

1) logger.save(..., livetime=36) — "срок жизни" файла лога в ЧАСАХ.
   Никаких фоновых процессов: при КАЖДОЙ записи проверяем, сколько живёт файл.
   Если возраст > livetime, файл ротируется (перезаписывается с нуля).
   Возраст берём из мета-файла <save_path>.meta (храним "время рождения" лога).
   Если мета-файл отсутствует, используем текущее время модификации файла как старт
   и создаём мету.

2) Флаг nl=True/False — управляет переносом строки:
   - По умолчанию nl=True — добавляется конец строки (или end, если задан).
   - Если nl=False — конец строки принудительно заменяется на пробел ' ' (удобно собирать
     разноцветные фрагменты в одной строке). Параметр end игнорируется, когда nl=False.
"""

from __future__ import annotations

import os
import sys
import time as _time
import threading
import traceback
from datetime import datetime
from typing import Optional, Sequence


# Печатать traceback автоматически при вызове красного лога внутри except
AUTO_TRACEBACK_ON_RED: bool = True



# Поддержка Windows-консоли (цвета)
try:
    import colorama  # type: ignore

    colorama.just_fix_windows_console()
    _COLORAMA_OK = True
except Exception:
    _COLORAMA_OK = False


def _now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Строка текущего локального времени по формату strftime."""
    return datetime.now().strftime(fmt)


class MPrintLogger:
    """
    Класс цветного логгера с методами-цветами (red/green/...).
    Методы формируются динамически по словарю PALETTE.
    """

    # Имя -> ANSI SGR код (foreground)
    PALETTE = {
        "black": 30,
        "red": 31,
        "green": 32,
        "yellow": 33,
        "blue": 34,
        "magenta": 35,
        "cyan": 36,
        "white": 37,
        "grey": 90,  # яркий "чёрный" (тёмно-серый)
        "bright_red": 91,
        "bright_green": 92,
    }

    ANSI_RESET = "\033[0m"
    ANSI_BOLD = "\033[1m"

    def __init__(self) -> None:
        # Настройки форматирования
        self._time_format: str = "%Y-%m-%d %H:%M:%S"
        self._force_color: Optional[bool] = None  # None=auto по TTY, True/False — принудительно

        # ДЕФОЛТНОЕ ВРЕМЯ (set_default_time)
        self._default_time: bool = False
        self._default_time_file_only: bool = False

        # Состояние сохранения в файл
        self._save_enabled: bool = False
        self._save_path: Optional[str] = None
        self._save_filter: Optional[set[str]] = None  # None -> сохраняем всё; set -> только указанные
        self._save_mode: str = "a"

        # Ротация/удержание
        self._save_livetime_hours: Optional[int] = None  # через сколько часов ротировать активный файл
        self._save_keep_old_hours: Optional[int] = None  # через сколько часов удалять архивы
        self._save_rotate_unit: str = "hour"  # "hour" | "day"

        # Метаданные активного файла
        self._save_meta_path: Optional[str] = None  # путь к мета-файлу (рядом с активным логом)
        self._save_born_ts: Optional[float] = None  # "время рождения" активного лога (epoch sec)

        # Синхронизация ввода/вывода
        self._io_lock = threading.Lock()

    # ---------------- Публичные сервисные методы ----------------

    def save(
        self,
        save_path: str,
        colors: Optional[Sequence[str]] = None,
        mode: str = "a",
        ensure_dir: bool = True,
        livetime: Optional[int] = None,
        keep_old: Optional[int] = None,
        rotate_unit: str = "hour",
    ) -> None:
        """
        Включить запись в файл.

        :param save_path: путь к АКТИВНОМУ файлу лога (не директория!).
        :param colors: список имён цветов для записи; если None или пусто — пишем все.
        :param mode: режим записи активного файла ('a' обычно). 'w' перезатирает при каждой записи.
        :param ensure_dir: создать каталоги при необходимости.
        :param livetime: срок жизни АКТИВНОГО файла в ЧАСАХ. Если None или <1 — ротация отключена.
        :param keep_old: хранить архивы ещё N часов, затем удалять. Если None или <1 — не удалять.
        :param rotate_unit: "hour" или "day" — формат суффикса архивного имени.
        """
        if not isinstance(save_path, str) or not save_path.strip():
            raise ValueError("save_path должен быть непустой строкой.")
        save_path = save_path.strip()

        rotate_unit = (rotate_unit or "").strip().lower()
        if rotate_unit not in ("hour", "day"):
            raise ValueError("rotate_unit должен быть 'hour' или 'day'.")

        if ensure_dir:
            dir_ = os.path.dirname(os.path.abspath(save_path))
            if dir_ and not os.path.exists(dir_):
                os.makedirs(dir_, exist_ok=True)

        self._save_path = os.path.abspath(save_path)
        self._save_mode = mode
        self._save_enabled = True
        self._save_rotate_unit = rotate_unit

        # Фильтр по цветам
        if colors:
            normalized = {c.strip().lower() for c in colors if isinstance(c, str) and c.strip()}
            self._save_filter = normalized or None
        else:
            self._save_filter = None  # сохраняем все

        # Ротация активного файла
        if isinstance(livetime, int) and livetime >= 1:
            self._save_livetime_hours = livetime
        else:
            self._save_livetime_hours = None

        # Удаление архивов
        if isinstance(keep_old, int) and keep_old >= 1:
            self._save_keep_old_hours = keep_old
        else:
            self._save_keep_old_hours = None

        # Метаданные активного лога (всегда рядом с активным файлом)
        self._save_meta_path = f"{self._save_path}.meta"
        self._save_born_ts = None  # будет инициализировано при первой записи

    def disable_save(self) -> None:
        """Выключить запись в файл и очистить связанные настройки."""
        self._save_enabled = False
        self._save_path = None
        self._save_filter = None
        self._save_mode = "a"

        self._save_livetime_hours = None
        self._save_keep_old_hours = None
        self._save_rotate_unit = "hour"

        self._save_meta_path = None
        self._save_born_ts = None

    def set_time_format(self, fmt: str) -> None:
        """Изменить формат метки времени (strftime)."""
        try:
            _ = datetime.now().strftime(fmt)
        except Exception as exc:
            raise ValueError(f"Некорректный формат времени: {fmt!r}") from exc
        self._time_format = fmt

    def set_default_time(self, enable: bool, *, file_only: bool = False) -> None:
        """
        Включить/выключить метку времени по умолчанию.

        Если в вызове НЕ передали time=..., то используется этот дефолт.

        :param enable: True/False — включить/выключить дефолт.
        :param file_only: если True, время добавляется только в файл (при save),
                         а в терминал — без времени (когда time не задан явно).
        """
        self._default_time = bool(enable)
        self._default_time_file_only = bool(file_only)

    def force_color(self, enable: Optional[bool]) -> None:
        """
        Принудительно управлять цветом вывода.
        :param enable: True/False — принудительно; None — авто по isatty().
        """
        if enable not in (True, False, None):
            raise ValueError("enable должен быть True, False или None.")
        self._force_color = enable

    # ---------------- Базовая реализация вывода ----------------

    def _format_message(self, args: tuple, sep: str) -> str:
        """
        Собрать текст сообщения.
        Поддерживает printf-стиль:
            red("x=%s y=%.2f", a, b)
        Если форматирование не удалось — fallback на join через sep.
        """
        if not args:
            return ""

        if len(args) == 1:
            return str(args[0])

        fmt = args[0]
        rest = args[1:]

        if isinstance(fmt, str):
            try:
                # Именованные плейсхолдеры: "%(k)s"
                if len(rest) == 1 and isinstance(rest[0], dict):
                    return fmt % rest[0]

                # Обычные: "%s", "%.12f" и т.п.
                if len(rest) == 1:
                    return fmt % rest[0]
                return fmt % tuple(rest)
            except Exception:
                pass

        return sep.join(map(str, args))

    def log(
        self,
        color: str,
        *args,
        b: bool = False,
        time: Optional[bool] = None,
        nl: bool = True,
        sep: str = " ",
        end: str = "\n",
        stream=None,
        flush: bool = True,
    ) -> None:
        """
        Универсальный метод печати в заданном цвете (и опционально жирным).

        time:
        - True/False: принудительно включить/выключить метку времени
        - None: использовать дефолт из set_default_time(...)
        """
        stream = sys.stdout if stream is None else stream

        # Сообщение (printf-стиль + fallback как print)
        msg = self._format_message(args, sep=sep)

        # Решаем, нужно ли время (терминал/файл)
        if time is None:
            time_for_file = self._default_time
            time_for_term = self._default_time and (not self._default_time_file_only)
        else:
            time_for_file = bool(time)
            time_for_term = bool(time)

        prefix_term = f"[{_now_str(self._time_format)}] " if time_for_term else ""
        prefix_file = f"[{_now_str(self._time_format)}] " if time_for_file else ""

        plain_text_term = f"{prefix_term}{msg}"
        plain_text_file = f"{prefix_file}{msg}"

        # Решаем — красить ли вывод
        use_color = self._should_colorize_stream(stream)
        color_name = (color or "").strip().lower()

        if use_color and color_name in self.PALETTE:
            sgr = self._compose_sgr(self.PALETTE[color_name], bold=b)
            term_text = f"{prefix_term}{sgr}{msg}{self.ANSI_RESET}"
        else:
            term_text = plain_text_term

        # Итоговый "конец строки" (только для терминала)
        end_text = " " if (nl is False) else ("" if end is None else end)

        # Нужно ли автодопечатывать traceback
        need_tb = False
        tb_text = ""
        if AUTO_TRACEBACK_ON_RED and color_name == "red":
            exc_type, _, _ = sys.exc_info()
            if exc_type is not None:
                tb_text = traceback.format_exc()
                tb_text = (tb_text or "").rstrip("\n")
                if tb_text:
                    need_tb = True

        with self._io_lock:
            # Печать основного сообщения в поток
            try:
                stream.write(term_text + end_text)
            except Exception:
                stream.write(plain_text_term + end_text)

            if flush:
                try:
                    stream.flush()
                except Exception:
                    pass

            # Запись основного сообщения в файл (без ANSI)
            self._maybe_save(color_name, plain_text_file)

            # Авто-traceback (только если мы реально внутри except)
            if need_tb:
                # Гарантируем, что traceback начнётся с новой строки
                ended_with_newline = bool(end_text) and (end_text.endswith("\n") or end_text.endswith("\r"))
                if not ended_with_newline:
                    try:
                        stream.write("\n")
                    except Exception:
                        pass

                tb_plain_term = f"{prefix_term}{tb_text}"
                tb_plain_file = f"{prefix_file}{tb_text}"

                if use_color and color_name in self.PALETTE:
                    sgr_tb = self._compose_sgr(self.PALETTE[color_name], bold=False)
                    tb_term_text = f"{prefix_term}{sgr_tb}{tb_text}{self.ANSI_RESET}"
                else:
                    tb_term_text = tb_plain_term

                try:
                    stream.write(tb_term_text + "\n")
                except Exception:
                    try:
                        stream.write(tb_plain_term + "\n")
                    except Exception:
                        pass

                if flush:
                    try:
                        stream.flush()
                    except Exception:
                        pass

                # И в файл тоже (отдельной записью)
                self._maybe_save(color_name, tb_plain_file)


    # ---------------- Внутренние утилиты ----------------

    def _compose_sgr(self, fg_code: int, bold: bool) -> str:
        """Собрать ANSI-последовательность: жирный (опц.) + цвет."""
        parts = []
        if bold:
            parts.append(self.ANSI_BOLD)
        parts.append(f"\033[{fg_code}m")
        return "".join(parts)

    def _should_colorize_stream(self, stream) -> bool:
        """Определить, стоит ли красить вывод в указанный поток."""
        if self._force_color is True:
            return True
        if self._force_color is False:
            return False
        try:
            return bool(getattr(stream, "isatty", lambda: False)())
        except Exception:
            return False

    # ----------- Лог-метаданные и ротация по времени жизни -----------

    def _ensure_log_birth(self) -> None:
        """
        Инициализировать "время рождения" лога (_save_born_ts) и мета-файл:
        - если мета есть — читаем его;
        - если меты нет, но файл существует — берём mtime файла и создаём мету;
        - если файла нет — фиксируем текущее время и создаём мету.
        """
        if not self._save_enabled or not self._save_path or not self._save_meta_path:
            return
        if self._save_born_ts is not None:
            return

        try:
            if os.path.exists(self._save_meta_path):
                ts = self._read_meta_ts(self._save_meta_path)
                self._save_born_ts = ts if ts is not None else _time.time()
            elif os.path.exists(self._save_path):
                ts = os.path.getmtime(self._save_path)
                self._save_born_ts = ts
                self._write_meta_ts(self._save_meta_path, ts)
            else:
                now = _time.time()
                self._save_born_ts = now
                self._write_meta_ts(self._save_meta_path, now)
        except Exception:
            self._save_born_ts = _time.time()

    def _read_meta_ts(self, meta_path: str) -> Optional[float]:
        """Прочитать epoch-timestamp из мета-файла; вернуть None при ошибке."""
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                raw = fh.read().strip()
            if not raw:
                return None
            return float(raw)
        except Exception:
            return None

    def _write_meta_ts(self, meta_path: str, ts: float) -> None:
        """Записать epoch-timestamp в мета-файл (тихо игнорировать ошибки)."""
        try:
            with open(meta_path, "w", encoding="utf-8") as fh:
                fh.write(str(int(ts)))
        except Exception:
            pass

    def _archive_suffix(self, ts: float) -> str:
        dt = datetime.fromtimestamp(ts)  # локальное время
        if self._save_rotate_unit == "day":
            return dt.strftime("%Y%m%d")
        return dt.strftime("%Y%m%d_%H")

    def _build_archive_path(self, born_ts: float) -> str:
        """
        Из logs/manager.log делаем logs/manager_YYYYMMDD_HH.log (или ..._YYYYMMDD.log)
        """
        assert self._save_path is not None
        base_dir = os.path.dirname(self._save_path)
        base_name = os.path.basename(self._save_path)

        root, ext = os.path.splitext(base_name)
        suffix = self._archive_suffix(born_ts)

        archive_name = f"{root}_{suffix}{ext}"
        return os.path.join(base_dir, archive_name)

    def _safe_rename_to_archive(self, src_path: str, dst_path: str) -> None:
        """
        Переименовать src -> dst.
        Если dst уже существует, добавляем _001, _002, ...
        """
        if not os.path.exists(src_path):
            return

        # если файл пустой — нет смысла архивировать
        try:
            if os.path.getsize(src_path) == 0:
                return
        except Exception:
            pass

        root, ext = os.path.splitext(dst_path)
        candidate = dst_path

        for i in range(0, 1000):
            if not os.path.exists(candidate):
                try:
                    os.rename(src_path, candidate)
                except Exception:
                    # fallback: копирование+удаление
                    try:
                        with open(src_path, "rb") as rf, open(candidate, "wb") as wf:
                            wf.write(rf.read())
                        os.remove(src_path)
                    except Exception:
                        pass
                return
            candidate = f"{root}_{i+1:03d}{ext}"

    def _cleanup_old_archives(self) -> None:
        """
        Удалить архивные файлы старше keep_old часов.
        Архивы распознаём по имени: <root>_<YYYYMMDD(_HH)><ext>
        """
        if not self._save_enabled or not self._save_path:
            return
        if self._save_keep_old_hours is None:
            return

        base_dir = os.path.dirname(self._save_path)
        base_name = os.path.basename(self._save_path)
        root, ext = os.path.splitext(base_name)

        prefix = f"{root}_"
        now = _time.time()

        try:
            for name in os.listdir(base_dir or "."):
                if name == base_name:
                    continue
                if not name.startswith(prefix) or not name.endswith(ext):
                    continue

                full_path = os.path.join(base_dir, name)
                try:
                    age_hours = (now - os.path.getmtime(full_path)) / 3600.0
                    if age_hours > float(self._save_keep_old_hours):
                        try:
                            os.remove(full_path)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _rotate_if_needed(self) -> None:
        """
        Проверить срок жизни активного файла и ротировать при необходимости.

        Ротация:
        - rename(save_path -> archive)
        - truncate/create new save_path
        - update meta born_ts
        """
        if not self._save_enabled or not self._save_path:
            return

        # Даже если ротация выключена — можем чистить архивы по keep_old
        if self._save_livetime_hours is None:
            self._cleanup_old_archives()
            return

        self._ensure_log_birth()
        if self._save_born_ts is None:
            self._cleanup_old_archives()
            return

        age_hours = (_time.time() - self._save_born_ts) / 3600.0
        if age_hours > float(self._save_livetime_hours):
            # 1) архивируем текущий файл (если он есть)
            archive_path = self._build_archive_path(self._save_born_ts)
            self._safe_rename_to_archive(self._save_path, archive_path)

            # 2) создаём/очищаем новый активный файл
            try:
                with open(self._save_path, "w", encoding="utf-8"):
                    pass
            except Exception:
                pass

            # 3) обновляем born_ts + meta
            now = _time.time()
            self._save_born_ts = now
            if self._save_meta_path:
                self._write_meta_ts(self._save_meta_path, now)

        # 4) чистим архивы по keep_old
        self._cleanup_old_archives()

    def _maybe_save(self, color_name: str, plain_text: str) -> None:
        """Сохранить строку в файл (если включено и цвет проходит фильтр)."""
        if not self._save_enabled or not self._save_path:
            return
        if self._save_filter and color_name not in self._save_filter:
            return

        # Проверка срока жизни и возможная ротация
        self._rotate_if_needed()

        # Запись строки (в файл всегда добавляем перевод строки)
        try:
            with open(self._save_path, self._save_mode, encoding="utf-8") as fh:
                fh.write(plain_text + "\n")
        except Exception:
            pass


# ---------- Динамическая генерация методов по цветам (внутри класса) ----------

def _make_color_method(color_name: str):
    def method(
        self: MPrintLogger,
        *args,
        b: bool = False,
        time: Optional[bool] = None,
        nl: bool = True,
        sep: str = " ",
        end: str = "\n",
        stream=None,
        flush: bool = True,
    ) -> None:
        return self.log(
            color_name,
            *args,
            b=b,
            time=time,
            nl=nl,
            sep=sep,
            end=end,
            stream=stream,
            flush=flush,
        )

    method.__name__ = color_name
    return method


for _c in list(MPrintLogger.PALETTE.keys()):
    setattr(MPrintLogger, _c, _make_color_method(_c))


# ---------- Экземпляр для импорта (как и раньше) ----------

logger = MPrintLogger()


# ---------- Прямые функции-обёртки (уровень модуля) ----------

def _make_color_function(color_name: str):
    """
    Сгенерировать функцию уровня модуля для цвета.
    """
    def func(
        *args,
        b: bool = False,
        time: Optional[bool] = None,
        nl: bool = True,
        sep: str = " ",
        end: str = "\n",
        stream=None,
        flush: bool = True,
    ) -> None:
        return getattr(logger, color_name)(
            *args,
            b=b,
            time=time,
            nl=nl,
            sep=sep,
            end=end,
            stream=stream,
            flush=flush,
        )

    func.__name__ = color_name
    func.__doc__ = (
        f"Функция вывода в цвете '{color_name}' без обращения к logger. "
        "Параметры: b=True, time=True/False/None, nl=True/False, sep, end, stream, flush."
    )
    return func


# Эти функции хотим иметь как обычные def, чтобы IDE подсвечивал/автокомплитил
_STATIC_COLOR_FUNCS = {"green", "red", "blue"}


def green(
    *args,
    b: bool = False,
    time: Optional[bool] = None,
    nl: bool = True,
    sep: str = " ",
    end: str = "\n",
    stream=None,
    flush: bool = True,
) -> None:
    """Функция вывода в цвете 'green' без обращения к logger."""
    return logger.green(*args, b=b, time=time, nl=nl, sep=sep, end=end, stream=stream, flush=flush)


def red(
    *args,
    b: bool = False,
    time: Optional[bool] = None,
    nl: bool = True,
    sep: str = " ",
    end: str = "\n",
    stream=None,
    flush: bool = True,
) -> None:
    """Функция вывода в цвете 'red' без обращения к logger."""
    return logger.red(*args, b=b, time=time, nl=nl, sep=sep, end=end, stream=stream, flush=flush)


def blue(
    *args,
    b: bool = False,
    time: Optional[bool] = None,
    nl: bool = True,
    sep: str = " ",
    end: str = "\n",
    stream=None,
    flush: bool = True,
) -> None:
    """Функция вывода в цвете 'blue' без обращения к logger."""
    return logger.blue(*args, b=b, time=time, nl=nl, sep=sep, end=end, stream=stream, flush=flush)


# Создаём функции остальных цветов динамически (НЕ перетирая green/red/blue)
for _c in list(MPrintLogger.PALETTE.keys()):
    if _c in _STATIC_COLOR_FUNCS:
        continue
    globals()[_c] = _make_color_function(_c)


# ---------- Прямые обёртки над сервисными методами logger ----------

def log(
    color: str,
    *args,
    b: bool = False,
    time: Optional[bool] = None,
    nl: bool = True,
    sep: str = " ",
    end: str = "\n",
    stream=None,
    flush: bool = True,
) -> None:
    """Прямой вызов logger.log(...)."""
    return logger.log(color, *args, b=b, time=time, nl=nl, sep=sep, end=end, stream=stream, flush=flush)


def save(
    save_path: str,
    colors: Optional[Sequence[str]] = None,
    mode: str = "a",
    ensure_dir: bool = True,
    livetime: Optional[int] = None,
    keep_old: Optional[int] = None,
    rotate_unit: str = "hour",
) -> None:
    """Прямой вызов logger.save(...)."""
    return logger.save(
        save_path,
        colors=colors,
        mode=mode,
        ensure_dir=ensure_dir,
        livetime=livetime,
        keep_old=keep_old,
        rotate_unit=rotate_unit,
    )


def disable_save() -> None:
    """Прямой вызов logger.disable_save()."""
    return logger.disable_save()


def set_time_format(fmt: str) -> None:
    """Прямой вызов logger.set_time_format(...)."""
    return logger.set_time_format(fmt)


def set_default_time(enable: bool, *, file_only: bool = False) -> None:
    """Прямой вызов logger.set_default_time(...)."""
    return logger.set_default_time(enable, file_only=file_only)


def force_color(enable: Optional[bool]) -> None:
    """Прямой вызов logger.force_color(...)."""
    return logger.force_color(enable)


def row(
    color_map: dict[str, object],
    *,
    sep: str = " ",
    stream=None,
    flush: bool = True,
    bold: bool = False,
) -> None:
    """
    Напечатать значения в ОДНУ строку (без перевода строки), раскрашивая по ключам словаря.

    Пример:
        d = {"green": 123, "red": "ERR", "yellow": 0.25}
        row(d)  # напечатает: 123 ERR 0.25 (каждое своим цветом), без \n в конце

    :param color_map: dict вида { "green": value1, "red": value2, ... }
                      Порядок вывода = порядок элементов dict (в Python 3.7+ он сохраняется).
    :param sep: разделитель между фрагментами.
    :param stream: куда печатать (по умолчанию sys.stdout).
    :param flush: делать flush потока.
    :param bold: печатать все фрагменты жирным (True/False).
    """
    if not color_map:
        return

    stream = sys.stdout if stream is None else stream

    # Решаем, красить ли вывод (учитывает logger.force_color(...) и isatty())
    use_color = False
    try:
        use_color = logger._should_colorize_stream(stream)  # noqa: SLF001
    except Exception:
        use_color = False

    parts: list[str] = []
    for color_name, value in color_map.items():
        text = str(value)
        c = (color_name or "").strip().lower()

        if use_color and c in MPrintLogger.PALETTE:
            try:
                sgr = logger._compose_sgr(MPrintLogger.PALETTE[c], bold=bold)  # noqa: SLF001
                parts.append(f"{sgr}{text}{MPrintLogger.ANSI_RESET}")
            except Exception:
                parts.append(text)
        else:
            parts.append(text)

    try:
        stream.write(sep.join(parts))  # ВАЖНО: без перевода строки
    except Exception:
        # Совсем аварийный фоллбек
        try:
            sys.stdout.write(sep.join(parts))
        except Exception:
            return

    if flush:
        try:
            stream.flush()
        except Exception:
            pass


def _logger_row(
    self: MPrintLogger,
    color_map: dict[str, object],
    *,
    sep: str = " ",
    stream=None,
    flush: bool = True,
    bold: bool = False,
) -> None:
    """То же самое, но как метод logger.row(...)."""
    return row(color_map, sep=sep, stream=stream, flush=flush, bold=bold)


# Добавляем как метод экземплярам/классу (ничего не меняем в общей логике логгера)
setattr(MPrintLogger, "row", _logger_row)


# Явно объявим публичный API модуля (для удобства autocomplete и from mprint import *)
__all__ = (
    ["logger", "log", "save", "disable_save", "set_time_format", "set_default_time", "force_color"]
    + list(MPrintLogger.PALETTE.keys())
)
