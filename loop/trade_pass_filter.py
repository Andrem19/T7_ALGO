from __future__ import annotations

from typing import List, Tuple


def calc_long_short_v2(
    *,
    dow: int,
    atr: int,
    rsi: float,
    iv_est: float,
    hill: int,
    cl_4h: int
) -> Tuple[bool, bool, str]:
    """
    Полный эквивалент логики 1:

        if dow in (0, 1, 3, 4, 6) and atr in (1, 2) and rsi < 76:
            if dow in (0, 6):
                return 0
            return 2
        if dow in (0, 1, 2, 4, 6) and atr in (0, 3, 4) and iv_est > 0.33 and rsi > 31:
            return 1
        return 0

    Возвращает:
      long=True  <=> return 1
      short=True <=> return 2
      иначе return 0

    Важно: при срабатывании RULE1 и dow in {0,6} идёт ранний "return 0"
    (то есть short НЕ включается).
    """

    rule1_fail: List[str] = []
    rule2_fail: List[str] = []

    # ---------------- RULE 1 ----------------
    rule1_ok = (dow in {0, 1, 3, 4, 6}) and (atr in {1, 2}) and (rsi < 76) and (hill != 1) and (cl_4h != 0)

    if rule1_ok:
        # В логике 1 тут ранний возврат (0 или 2), RULE2 уже не проверяется.
        if dow in {0, 6}:
            explanation = (
                "RESULT_CODE: 0 | RULE1: PASS, но dow в {0, 6} → ранний return 0 "
                "| RULE2: SKIPPED"
            )
            return False, False, explanation

        # dow = 1,3,4 => return 2
        explanation = (
            "RESULT_CODE: 2 | RULE1: PASS, dow не в {0, 6} → return 2 (short=True) "
            "| RULE2: SKIPPED"
        )
        return False, True, explanation

    # Если RULE1 не прошло — собираем причины (для объяснения)
    if dow not in {0, 1, 3, 4, 6}:
        rule1_fail.append("dow не в {0, 1, 3, 4, 6}")
    if atr not in {1, 2}:
        rule1_fail.append("atr не в {1, 2}")
    if not (rsi < 76):
        rule1_fail.append("rsi не меньше 76")
    if hill == 1:
        rule1_fail.append("hill in 1")
    if cl_4h == 0:
        rule1_fail.append("cl_4h in 0")

    # ---------------- RULE 2 ----------------
    rule2_ok = (dow in {0, 1, 2, 4, 6}) and (atr in {0, 3, 4}) and (iv_est > 0.33) and (rsi > 31)

    if rule2_ok:
        explanation = (
            "RESULT_CODE: 1 | RULE1: FAIL — " + "; ".join(rule1_fail) +
            " | RULE2: PASS → return 1 (long=True)"
        )
        return True, False, explanation

    # RULE2 тоже не прошло — причины
    if dow not in {0, 1, 2, 4, 6}:
        rule2_fail.append("dow не в {0, 1, 2, 4, 6}")
    if atr not in {0, 3, 4}:
        rule2_fail.append("atr не в {0, 3, 4}")
    if not (iv_est > 0.33):
        rule2_fail.append("iv_est не больше 0.33")
    if not (rsi > 31):
        rule2_fail.append("rsi не больше 31")

    explanation = (
        "RESULT_CODE: 0 | RULE1: FAIL — " + "; ".join(rule1_fail) +
        " | RULE2: FAIL — " + "; ".join(rule2_fail) +
        " → return 0"
    )
    return False, False, explanation




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
    # if not same:
    #     short_fail.append("reg_d != ch_1d (режимы не совпали)")
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



def calc_long_short_v3(
    *,
    dow: int,
    atr: int,
    rsi: float,
    iv_est: float,
    hill: int,
    cl_4h: int
) -> Tuple[bool, bool, str]:
    long_fail: List[str] = []
    short_fail: List[str] = []
    
    
    
    
    if dow not in [1,3,4]:
        short_fail.append('dow not in [1,3,4]')
    if atr not in [1,2]:
        short_fail.append('atr not in [1,2]')
    if rsi > 76:
        short_fail.append('rsi > 76')
    if hill == 1:
        short_fail.append('hill == 1')
    if cl_4h == 0:
        short_fail.append('cl_4h == 0')
        
        
    short = (len(short_fail) == 0)
        
    
    if dow not in [0,1,2,4,6]:
        long_fail.append('dow not in [0,1,2,4,6]')
    if atr not in [0, 3, 4]:
        long_fail.append('atr not in [0, 3, 4]')
    if iv_est < 0.33:
        long_fail.append('iv_est < 0.33')
    if rsi < 31:
        long_fail.append('rsi < 31')
    
    
    long = (len(long_fail) == 0)
    
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

    explanation = f"{result_txt}\n{long_txt}\n{short_txt}"
    return long, short, explanation


def calc_long_short_v4(
    vars_1: dict,
    rules_1: dict,
    rules_2: dict
) -> Tuple[bool, bool, str]:
    # print(f'{vars_1}\n{rules_1}\n{rules_2}')
    try:
        long_fail: List[str] = []
        short_fail: List[str] = []
        
        for k, v in rules_2.items():
            if type(v) is list:
                if vars_1[k] not in v:
                    short_fail.append(f'{k} not in {v}')
            if type(v) is str:
                lower = True if v[:1] == '<' else False if v[:1] == '>' else None
                if lower is None:
                    return False, False, f'Something went wrong {v}'
                elif lower and vars_1[k] > float(v[1:]):
                    short_fail.append(f'{k} > {v[1:]}')
                elif not lower and vars_1[k] < float(v[1:]):
                    short_fail.append(f'{k} < {v[1:]}')
        
        short = (len(short_fail) == 0)
        
        for k, v in rules_1.items():
            if type(v) is list:
                if vars_1[k] not in v:
                    long_fail.append(f'{k} not in {v}')
            if type(v) is str:
                lower = True if v[:1] == '<' else False if v[:1] == '>' else None
                if lower is None:
                    return False, False, f'Something went wrong {v}'
                elif lower and vars_1[k] > float(v[1:]):
                    long_fail.append(f'{k} > {v[1:]}')
                elif not lower and vars_1[k] < float(v[1:]):
                    long_fail.append(f'{k} < {v[1:]}')
        
        
        long = (len(long_fail) == 0)
        
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

        explanation = f"{result_txt}\n{long_txt}\n{short_txt}"
        return long, short, explanation    
    except Exception as e:
        return False, False, f'Something went wrong\n{e}\n{vars_1}'
    