from __future__ import annotations

from typing import List, Tuple


def calc_long_short(
    *,
    dow: int,
    ch_1d: int,
    reg_d: int,
    reg_h: int,
) -> Tuple[Tuple[bool, bool], str]:
    """
    Возвращает:
      1) (long, short) — булевы флаги
      2) explanation — строка с объяснением, что прошло/не прошло и почему
    """

    same = (reg_d == ch_1d)

    long_fail: List[str] = []
    short_fail: List[str] = []

    # --- LONG checks (всё в одной логике, но причины собираем отдельно) ---
    if not same:
        long_fail.append("reg_d != ch_1d (режимы не совпали)")
    if ch_1d == 1:
        long_fail.append("ch_1d == 1 (лонг запрещён)")
    if dow in {3, 5}:
        long_fail.append("dow в {3, 5} (день недели запрещён для лонга)")
    if dow == 4 and ch_1d == 0:
        long_fail.append("особое правило: dow == 4 и ch_1d == 0 (лонг запрещён)")
    if reg_h == 4:
        long_fail.append("reg_h == 4 (часовой режим запрещён для лонга)")
    if not (ch_1d == 2 or dow == 2):
        long_fail.append("нужно: ch_1d == 2 ИЛИ dow == 2")

    long = (len(long_fail) == 0)

    # --- SHORT checks ---
    if not same:
        short_fail.append("reg_d != ch_1d (режимы не совпали)")
    if ch_1d == 2:
        short_fail.append("ch_1d == 2 (шорт запрещён)")
    if dow in {2, 5, 6}:
        short_fail.append("dow в {2, 5, 6} (день недели запрещён для шорта)")
    if reg_h in {0, 4, 5}:
        short_fail.append("reg_h в {0, 4, 5} (часовой режим запрещён для шорта)")
    if not (ch_1d == 1 or dow == 3):
        short_fail.append("нужно: ch_1d == 1 ИЛИ dow == 3")

    short = (len(short_fail) == 0)

    # --- Explanation ---
    if long:
        long_txt = "LONG: PASS"
    else:
        long_txt = "LONG: FAIL — " + "; ".join(long_fail)

    if short:
        short_txt = "SHORT: PASS"
    else:
        short_txt = "SHORT: FAIL — " + "; ".join(short_fail)

    if long and not short:
        result_txt = "RESULT: long=True, short=False (вход в LONG)"
    elif short and not long:
        result_txt = "RESULT: long=False, short=True (вход в SHORT)"
    elif long and short:
        result_txt = "RESULT: long=True, short=True (одновременно) — проверьте правила"
    else:
        result_txt = "RESULT: long=False, short=False (входа нет)"

    explanation = f"{result_txt} | {long_txt} | {short_txt}"
    return long, short, explanation
