def get_comb_classes_1(x: dict) -> dict:
    """
    Кодирует 100 “ban-паттернов” в cl_1..cl_10.

    - Вход: один словарь с фичами (можно передать trade-словарь; тогда возьмём x["features"], если он есть).
    - Выход: {"cl_1": int, ..., "cl_10": int}

    Логика как раньше:
    - Внутри каждого cl_* правила идут сверху вниз по приоритету (сначала более специфичные, потом более общие).
    - Возвращается ПЕРВОЕ совпавшее правило => в одном cl_* выбирается максимум одно значение.
    """

    # если передали trade-словарь, а не голые фичи
    if isinstance(x.get("features"), dict):
        f = x["features"]
    else:
        f = x

    def _get_int(key: str):
        v = f.get(key, None)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return v

    def _match(rule: dict) -> bool:
        for k, need in rule.items():
            if _get_int(k) != need:
                return False
        return True

    def _eval_group(rules: list) -> int:
        """
        rules: list[tuple[str, dict]]
        Возвращает:
          0 если ничего не совпало, иначе 1..len(rules) по порядку.
        """
        for i, (_, cond) in enumerate(rules, 1):
            if _match(cond):
                return i
        return 0

    # =========================================================================
    # FIRST 50 RULES -> распределено по cl_1..cl_5
    # =========================================================================

    # cl_1: семейство cl_1d=0 (все его вариации)
    cl_1_rules = [
        ("cl_1d=0 & cl_1h=3 & squize_index=0", {"cl_1d": 0, "cl_1h": 3, "squize_index": 0}),  # #1
        ("cl_1d=0 & cl_1h=3", {"cl_1d": 0, "cl_1h": 3}),                                      # #2
        ("atr=1 & cl_1d=0 & squize_index=0", {"atr": 1, "cl_1d": 0, "squize_index": 0}),       # #40
        ("atr=1 & cl_1d=0", {"atr": 1, "cl_1d": 0}),                                           # #4
        ("atr_ratio_24h_7d=3 & cl_1d=0 & hill=0", {"atr_ratio_24h_7d": 3, "cl_1d": 0, "hill": 0}),  # #29
        ("cl_1d=0 & dvol_minus_rv_12h=3 & hill=0", {"cl_1d": 0, "dvol_minus_rv_12h": 3, "hill": 0}), # #44
        ("cl_1d=0 & dvol_minus_rv_12h=3", {"cl_1d": 0, "dvol_minus_rv_12h": 3}),               # #34
        ("cl_1d=0 & iv_est=2", {"cl_1d": 0, "iv_est": 2}),                                     # #11
        ("cl_1d=0 & rsi=4", {"cl_1d": 0, "rsi": 4}),                                           # #47
        ("cl_1d=0 & cls_15m=7", {"cl_1d": 0, "cls_15m": 7}),                                   # #37
        ("cl_15m=3 & cl_1d=0", {"cl_15m": 3, "cl_1d": 0}),                                     # #48
        ("cl_1d=0 & cls_30m=3", {"cl_1d": 0, "cls_30m": 3}),                                   # #41
        ("cl_1d=0", {"cl_1d": 0}),                                                              # #3
    ]
    cl_1 = _eval_group(cl_1_rules)

    # cl_2: семейство cl_1d=4
    cl_2_rules = [
        ("cl_1d=4 & d=1 & squize_index=0", {"cl_1d": 4, "d": 1, "squize_index": 0}),            # #8
        ("cl_1d=4 & d=1", {"cl_1d": 4, "d": 1}),                                                # #7
        ("cl_15m=3 & cl_1d=4 & squize_index=0", {"cl_15m": 3, "cl_1d": 4, "squize_index": 0}),  # #31
        ("cl_15m=3 & cl_1d=4", {"cl_15m": 3, "cl_1d": 4}),                                      # #23
        ("cl_1d=4 & cls_1h=0", {"cl_1d": 4, "cls_1h": 0}),                                      # #9
        ("cl_1d=4 & rsi=4", {"cl_1d": 4, "rsi": 4}),                                            # #16
    ]
    cl_2 = _eval_group(cl_2_rules)

    # cl_3: семейство iv_est=0 (где cl_1d=0/4 не требуется)
    cl_3_rules = [
        ("cl_15m=3 & iv_est=0 & squize_index=0", {"cl_15m": 3, "iv_est": 0, "squize_index": 0}),  # #26
        ("cl_15m=3 & iv_est=0", {"cl_15m": 3, "iv_est": 0}),                                      # #6
        ("atr_ratio_24h_7d=3 & iv_est=0", {"atr_ratio_24h_7d": 3, "iv_est": 0}),                  # #17
        ("d=0 & iv_est=0", {"d": 0, "iv_est": 0}),                                                # #30
        ("cl_1h=1 & iv_est=0", {"cl_1h": 1, "iv_est": 0}),                                        # #36
        ("atr=2 & iv_est=0", {"atr": 2, "iv_est": 0}),                                            # #39
        ("cl_15m=0 & h=6 & iv_est=0", {"cl_15m": 0, "h": 6, "iv_est": 0}),                        # #42
        ("iv_est=0", {"iv_est": 0}),                                                              # #19
    ]
    cl_3 = _eval_group(cl_3_rules)

    # cl_4: семейство squize_index=4 / dvol_minus_rv_12h=3 вокруг него
    cl_4_rules = [
        ("cl_15m=0 & dvol_minus_rv_12h=3 & hill=0 & squize_index=4",
         {"cl_15m": 0, "dvol_minus_rv_12h": 3, "hill": 0, "squize_index": 4}),                   # #32
        ("dvol_minus_rv_12h=3 & hill=0 & squize_index=4",
         {"dvol_minus_rv_12h": 3, "hill": 0, "squize_index": 4}),                                # #28
        ("dvol_minus_rv_12h=3 & squize_index=4", {"dvol_minus_rv_12h": 3, "squize_index": 4}),   # #18
        ("cl_15m=0 & dvol_minus_rv_12h=3 & squize_index=4",
         {"cl_15m": 0, "dvol_minus_rv_12h": 3, "squize_index": 4}),                              # #20
        ("atr_ratio_24h_7d=0 & cl_1h=2 & hill=0 & squize_index=4",
         {"atr_ratio_24h_7d": 0, "cl_1h": 2, "hill": 0, "squize_index": 4}),                     # #27
        ("cl_1h=2 & dvol_minus_rv_12h=3 & squize_index=4",
         {"cl_1h": 2, "dvol_minus_rv_12h": 3, "squize_index": 4}),                               # #45
        ("atr_ratio_24h_7d=0 & cl_1h=2 & squize_index=4",
         {"atr_ratio_24h_7d": 0, "cl_1h": 2, "squize_index": 4}),                                # #46
    ]
    cl_4 = _eval_group(cl_4_rules)

    # cl_5: остаток первого списка (прочее)
    cl_5_rules = [
        ("atr_ratio_24h_7d=4 & cl_15m=3 & rsi=4 & squize_index=0",
         {"atr_ratio_24h_7d": 4, "cl_15m": 3, "rsi": 4, "squize_index": 0}),                     # #14
        ("atr_ratio_24h_7d=4 & cl_15m=3 & rsi=4", {"atr_ratio_24h_7d": 4, "cl_15m": 3, "rsi": 4}), # #13
        ("cls_1h=7 & dvol_minus_rv_12h=4 & hill=0", {"cls_1h": 7, "dvol_minus_rv_12h": 4, "hill": 0}), # #25
        ("cls_1h=7 & dvol_minus_rv_12h=4", {"cls_1h": 7, "dvol_minus_rv_12h": 4}),               # #24
        ("atr_ratio_24h_7d=2 & cls_1h=7", {"atr_ratio_24h_7d": 2, "cls_1h": 7}),                 # #38
        ("cl_4h=2 & rsi=4", {"cl_4h": 2, "rsi": 4}),                                             # #21
        ("atr=2 & rsi=4", {"atr": 2, "rsi": 4}),                                                 # #22
        ("atr_ratio_24h_7d=2 & d=1", {"atr_ratio_24h_7d": 2, "d": 1}),                           # #12
        ("squize_index=0 & super_cls=0", {"squize_index": 0, "super_cls": 0}),                   # #5
        ("cl_1d=3 & iv_est=1", {"cl_1d": 3, "iv_est": 1}),                                       # #10
        ("iv_est=1 & super_cls=0", {"iv_est": 1, "super_cls": 0}),                               # #15
        ("cl_15m=1 & rsi=1", {"cl_15m": 1, "rsi": 1}),                                           # #43
        ("cl_4h=1 & hill=0", {"cl_4h": 1, "hill": 0}),                                           # #35
        ("cls_1h=0 & cls_30m=5", {"cls_1h": 0, "cls_30m": 5}),                                   # #49
        ("cls_1h=0", {"cls_1h": 0}),                                                             # #50
    ]
    cl_5 = _eval_group(cl_5_rules)

    # =========================================================================
    # SECOND 50 RULES -> распределено по cl_6..cl_10
    # =========================================================================

    # cl_6: семейство cl_1d=1
    cl_6_rules = [
        ("cl_1d=1 & hill=0 & squize_index=0", {"cl_1d": 1, "hill": 0, "squize_index": 0}),       # #12
        ("cl_1d=1 & hill=0", {"cl_1d": 1, "hill": 0}),                                           # #3
        ("cl_1d=1 & squize_index=0", {"cl_1d": 1, "squize_index": 0}),                           # #10
        ("cl_1d=1 & cl_4h=0 & squize_index=0", {"cl_1d": 1, "cl_4h": 0, "squize_index": 0}),     # #43
        ("cl_1d=1 & cl_4h=0", {"cl_1d": 1, "cl_4h": 0}),                                         # #15
        ("cl_1d=1 & cl_1h=3", {"cl_1d": 1, "cl_1h": 3}),                                         # #33
        ("cl_1d=1 & h=6 & hill=0", {"cl_1d": 1, "h": 6, "hill": 0}),                             # #38
        ("cl_1d=1 & h=6", {"cl_1d": 1, "h": 6}),                                                 # #41
        ("cl_1d=1", {"cl_1d": 1}),                                                               # #2
    ]
    cl_6 = _eval_group(cl_6_rules)

    # cl_7: семейство h=6 (без cl_1d=1 — оно уже в cl_6)
    cl_7_rules = [
        ("h=6 & hill=0 & squize_index=0", {"h": 6, "hill": 0, "squize_index": 0}),               # #23
        ("h=6 & hill=0 & ret_6h=2", {"h": 6, "hill": 0, "ret_6h": 2}),                           # #44
        ("cls_15m=7 & h=6 & hill=0", {"cls_15m": 7, "h": 6, "hill": 0}),                         # #37
        ("h=6 & ret_6h=2", {"h": 6, "ret_6h": 2}),                                               # #47
        ("h=6 & hill=0", {"h": 6, "hill": 0}),                                                   # #6
    ]
    cl_7 = _eval_group(cl_7_rules)

    # cl_8: семейство h=12
    cl_8_rules = [
        ("h=12 & hill=0", {"h": 12, "hill": 0}),                                                 # #46
        ("h=12 & iv_est=3", {"h": 12, "iv_est": 3}),                                             # #39
        ("cl_15m=0 & h=12", {"cl_15m": 0, "h": 12}),                                             # #40
        ("h=12", {"h": 12}),                                                                     # #45
    ]
    cl_8 = _eval_group(cl_8_rules)

    # cl_9: семейство iv_est=3 (кроме h=12 — оно в cl_8)
    cl_9_rules = [
        ("cl_1h=3 & iv_est=3 & squize_index=0", {"cl_1h": 3, "iv_est": 3, "squize_index": 0}),   # #48
        ("cl_1h=3 & iv_est=3", {"cl_1h": 3, "iv_est": 3}),                                       # #29
        ("iv_est=3 & squize_index=0", {"iv_est": 3, "squize_index": 0}),                         # #4
        ("iv_est=3", {"iv_est": 3}),                                                             # #5
    ]
    cl_9 = _eval_group(cl_9_rules)

    # cl_10: остаток второго списка (прочее)
    cl_10_rules = [
        ("atr=0 & cl_4h=4 & hill=0", {"atr": 0, "cl_4h": 4, "hill": 0}),                          # #49
        ("cl_4h=0 & hill=0 & squize_index=0", {"cl_4h": 0, "hill": 0, "squize_index": 0}),        # #50
        ("cl_4h=0 & hill=0", {"cl_4h": 0, "hill": 0}),                                            # #7
        ("cl_4h=0", {"cl_4h": 0}),                                                                # #26
        ("hill=0 & rsi=0 & squize_index=0", {"hill": 0, "rsi": 0, "squize_index": 0}),            # #27
        ("hill=0 & rsi=0", {"hill": 0, "rsi": 0}),                                                # #16
        ("atr=1 & hill=0 & squize_index=0", {"atr": 1, "hill": 0, "squize_index": 0}),            # #25
        ("atr_ratio_24h_7d=0 & hill=0 & squize_index=0", {"atr_ratio_24h_7d": 0, "hill": 0, "squize_index": 0}),  # #34
        ("atr=0 & atr_ratio_24h_7d=0 & hill=0", {"atr": 0, "atr_ratio_24h_7d": 0, "hill": 0}),    # #28
        ("atr=0 & hill=0", {"atr": 0, "hill": 0}),                                                # #13
        ("d=0 & hill=0", {"d": 0, "hill": 0}),                                                    # #21
        ("dvol_minus_rv_12h=1 & hill=0 & squize_index=0", {"dvol_minus_rv_12h": 1, "hill": 0, "squize_index": 0}),  # #42
        ("dvol_minus_rv_12h=1 & hill=0", {"dvol_minus_rv_12h": 1, "hill": 0}),                    # #22
        ("hill=0 & iv_est=2", {"hill": 0, "iv_est": 2}),                                          # #30
        ("cl_15m=2 & hill=0 & squize_index=0", {"cl_15m": 2, "hill": 0, "squize_index": 0}),      # #11
        ("cl_15m=2 & hill=0", {"cl_15m": 2, "hill": 0}),                                          # #32
        ("cl_15m=0 & squize_index=0", {"cl_15m": 0, "squize_index": 0}),                          # #14
        ("cl_15m=0", {"cl_15m": 0}),                                                              # #8
        ("cl_1h=2 & hill=0 & squize_index=0", {"cl_1h": 2, "hill": 0, "squize_index": 0}),        # #35
        ("cl_1h=2 & squize_index=0", {"cl_1h": 2, "squize_index": 0}),                            # #17
        ("cl_1h=2 & squize_index=0", {"cl_1h": 2, "squize_index": 0}),                            # #17 (дубликат по смыслу допустим; оставлен без вреда)
        ("cls_30m=7 & hill=0", {"cls_30m": 7, "hill": 0}),                                        # #18
        ("cls_30m=7", {"cls_30m": 7}),                                                            # #24
        ("cls_1h=3 & hill=0", {"cls_1h": 3, "hill": 0}),                                          # #36
        ("cls_1h=3 & squize_index=0", {"cls_1h": 3, "squize_index": 0}),                          # #31
        ("cls_1h=3", {"cls_1h": 3}),                                                              # #19
        ("cls_5m=6 & squize_index=0", {"cls_5m": 6, "squize_index": 0}),                          # #20
        ("cls_5m=6", {"cls_5m": 6}),                                                              # #9
        ("hill=0 & squize_index=0", {"hill": 0, "squize_index": 0}),                              # #1
    ]
    # уберём случайный дубль #17 внутри списка, чтобы индексы были стабильнее
    cl_10_rules = [
        r for i, r in enumerate(cl_10_rules)
        if not (i > 0 and r[0] == "cl_1h=2 & squize_index=0" and cl_10_rules[i - 1][0] == "cl_1h=2 & squize_index=0")
    ]
    cl_10 = _eval_group(cl_10_rules)

    return {
        "cl_1": cl_1,
        "cl_2": cl_2,
        "cl_3": cl_3,
        "cl_4": cl_4,
        "cl_5": cl_5,
        "cl_6": cl_6,
        "cl_7": cl_7,
        "cl_8": cl_8,
        "cl_9": cl_9,
        "cl_10": cl_10,
    }


def get_comb_classes_0(x: dict) -> dict:
    """
    Кодирует набор “ban-паттернов” в cl_1..cl_10 (по двум таблицам правил выше).

    - Вход: один словарь с фичами (можно передать trade-словарь; тогда возьмём x["features"], если он есть).
    - Выход: {"cl_1": int, ..., "cl_10": int}

    Внутри каждого cl_* правила идут сверху вниз по приоритету (сначала более специфичные).
    Возвращается ПЕРВОЕ совпавшее правило => внутри одного cl_* выбирается максимум одно значение.
    """

    # если передали trade-словарь, а не голые фичи
    if isinstance(x.get("features"), dict):
        f = x["features"]
    else:
        f = x

    def _get_int(key: str):
        v = f.get(key, None)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return v

    def _match(rule: dict) -> bool:
        for k, need in rule.items():
            if _get_int(k) != need:
                return False
        return True

    def _eval_group(rules: list) -> int:
        """
        rules: list[tuple[str, dict]]
        Возвращает:
          0 если ничего не совпало, иначе 1..len(rules) по порядку.
        """
        for i, (_, cond) in enumerate(rules, 1):
            if _match(cond):
                return i
        return 0

    # =========================================================================
    # FIRST TABLE (1..50) -> распределено по cl_1..cl_5
    # =========================================================================

    # cl_1: семейство cl_4h=2
    cl_1_rules = [
        ("cl_1d=1 & cl_4h=2 & squize_index=0", {"cl_1d": 1, "cl_4h": 2, "squize_index": 0}),  # #48
        ("cl_1d=1 & cl_4h=2", {"cl_1d": 1, "cl_4h": 2}),                                       # #47
        ("cl_4h=2 & dvol_minus_rv_12h=4", {"cl_4h": 2, "dvol_minus_rv_12h": 4}),               # #16
        ("cl_4h=2 & h=6", {"cl_4h": 2, "h": 6}),                                               # #36
        ("cl_4h=2 & hill=0", {"cl_4h": 2, "hill": 0}),                                         # #40
        ("cl_4h=2 & squize_index=0", {"cl_4h": 2, "squize_index": 0}),                         # #3
        ("cl_4h=2", {"cl_4h": 2}),                                                             # #1
    ]
    cl_1 = _eval_group(cl_1_rules)

    # cl_2: семейство atr_ratio_24h_7d=1 (все вариации)
    cl_2_rules = [
        ("atr_ratio_24h_7d=1 & hill=2 & ret_6h=0 & squize_index=0",
         {"atr_ratio_24h_7d": 1, "hill": 2, "ret_6h": 0, "squize_index": 0}),                  # #42
        ("atr_ratio_24h_7d=1 & hill=2 & ret_6h=0",
         {"atr_ratio_24h_7d": 1, "hill": 2, "ret_6h": 0}),                                     # #45
        ("atr_ratio_24h_7d=1 & hill=2 & squize_index=0",
         {"atr_ratio_24h_7d": 1, "hill": 2, "squize_index": 0}),                               # #22
        ("atr_ratio_24h_7d=1 & hill=2",
         {"atr_ratio_24h_7d": 1, "hill": 2}),                                                  # #29
        ("atr_ratio_24h_7d=1 & cl_1d=1 & hill=0",
         {"atr_ratio_24h_7d": 1, "cl_1d": 1, "hill": 0}),                                      # #21
        ("atr_ratio_24h_7d=1 & dvol_minus_rv_12h=4 & hill=0",
         {"atr_ratio_24h_7d": 1, "dvol_minus_rv_12h": 4, "hill": 0}),                          # #35
        ("atr_ratio_24h_7d=1 & h=6 & squize_index=0",
         {"atr_ratio_24h_7d": 1, "h": 6, "squize_index": 0}),                                  # #13
        ("atr_ratio_24h_7d=1 & ret_6h=0 & squize_index=0",
         {"atr_ratio_24h_7d": 1, "ret_6h": 0, "squize_index": 0}),                             # #25
        ("atr_ratio_24h_7d=1 & rsi=0 & squize_index=0",
         {"atr_ratio_24h_7d": 1, "rsi": 0, "squize_index": 0}),                                # #32
        ("atr_ratio_24h_7d=1 & dvol_minus_rv_12h=4",
         {"atr_ratio_24h_7d": 1, "dvol_minus_rv_12h": 4}),                                     # #5
        ("atr_ratio_24h_7d=1 & squize_index=0",
         {"atr_ratio_24h_7d": 1, "squize_index": 0}),                                          # #4
        ("atr_ratio_24h_7d=1 & h=6",
         {"atr_ratio_24h_7d": 1, "h": 6}),                                                     # #6
        ("atr_ratio_24h_7d=1 & cl_1h=1",
         {"atr_ratio_24h_7d": 1, "cl_1h": 1}),                                                 # #7
        ("atr_ratio_24h_7d=1 & cl_15m=1",
         {"atr_ratio_24h_7d": 1, "cl_15m": 1}),                                                # #11
        ("atr_ratio_24h_7d=1 & cls_1h=0",
         {"atr_ratio_24h_7d": 1, "cls_1h": 0}),                                                # #41
        ("atr_ratio_24h_7d=1 & ret_6h=0",
         {"atr_ratio_24h_7d": 1, "ret_6h": 0}),                                                # #19
        ("atr_ratio_24h_7d=1 & hill=0",
         {"atr_ratio_24h_7d": 1, "hill": 0}),                                                  # #20
        ("atr_ratio_24h_7d=1 & rsi=0",
         {"atr_ratio_24h_7d": 1, "rsi": 0}),                                                   # #28
        ("atr=3 & atr_ratio_24h_7d=1",
         {"atr": 3, "atr_ratio_24h_7d": 1}),                                                   # #33
        ("atr=2 & atr_ratio_24h_7d=1",
         {"atr": 2, "atr_ratio_24h_7d": 1}),                                                   # #44
        ("atr_ratio_24h_7d=1",
         {"atr_ratio_24h_7d": 1}),                                                             # #2
    ]
    cl_2 = _eval_group(cl_2_rules)

    # cl_3: семейство dvol_minus_rv_12h=4 (кроме cl_4h=2&dvol=4 — оно уже в cl_1)
    cl_3_rules = [
        ("dvol_minus_rv_12h=4 & h=6 & hill=0",
         {"dvol_minus_rv_12h": 4, "h": 6, "hill": 0}),                                         # #14
        ("dvol_minus_rv_12h=4 & hill=0",
         {"dvol_minus_rv_12h": 4, "hill": 0}),                                                 # #12
        ("dvol_minus_rv_12h=4",
         {"dvol_minus_rv_12h": 4}),                                                            # #9
    ]
    cl_3 = _eval_group(cl_3_rules)

    # cl_4: семейство d=4 / atr=4 / atr=2
    cl_4_rules = [
        ("atr=4 & cl_1d=1 & d=4", {"atr": 4, "cl_1d": 1, "d": 4}),                             # #31
        ("atr=4 & d=4 & squize_index=0", {"atr": 4, "d": 4, "squize_index": 0}),               # #15
        ("atr=4 & d=4", {"atr": 4, "d": 4}),                                                   # #17
        ("cl_1d=1 & d=4", {"cl_1d": 1, "d": 4}),                                               # #18
        ("d=4 & squize_index=0", {"d": 4, "squize_index": 0}),                                 # #34
        ("d=4", {"d": 4}),                                                                     # #49
        ("atr=2 & hill=0", {"atr": 2, "hill": 0}),                                             # #38
        ("atr=2", {"atr": 2}),                                                                 # #8
    ]
    cl_4 = _eval_group(cl_4_rules)

    # cl_5: прочее (одиночные/остаточные правила первого списка)
    cl_5_rules = [
        ("cl_1d=1 & h=6 & hill=0", {"cl_1d": 1, "h": 6, "hill": 0}),                           # #50
        ("cl_1h=1 & hill=0", {"cl_1h": 1, "hill": 0}),                                         # #30
        ("hill=0 & ret_6h=2", {"hill": 0, "ret_6h": 2}),                                       # #26
        ("ret_6h=2", {"ret_6h": 2}),                                                           # #10
        ("atr_ratio_24h_7d=2 & h=12", {"atr_ratio_24h_7d": 2, "h": 12}),                       # #39
        ("squize_index=4", {"squize_index": 4}),                                               # #23
        ("rsi=0", {"rsi": 0}),                                                                 # #24
        ("cls_30m=3", {"cls_30m": 3}),                                                         # #37
        ("cl_1d=4", {"cl_1d": 4}),                                                             # #46
        ("cl_1d=0", {"cl_1d": 0}),                                                             # #43
        ("cl_1h=1", {"cl_1h": 1}),                                                             # #27
    ]
    cl_5 = _eval_group(cl_5_rules)

    # =========================================================================
    # SECOND TABLE (1..50) -> распределено по cl_6..cl_10
    # =========================================================================

    # cl_6: семейство cl_4h=3 (включая atr=4 & cl_4h=3 ...)
    cl_6_rules = [
        ("atr=4 & cl_4h=3 & h=6 & iv_est=4 & squize_index=0",
         {"atr": 4, "cl_4h": 3, "h": 6, "iv_est": 4, "squize_index": 0}),                      # #50
        ("atr=4 & cl_4h=3 & h=6 & iv_est=4",
         {"atr": 4, "cl_4h": 3, "h": 6, "iv_est": 4}),                                        # #25
        ("atr=4 & cl_4h=3 & h=6 & squize_index=0",
         {"atr": 4, "cl_4h": 3, "h": 6, "squize_index": 0}),                                  # #5
        ("atr=4 & cl_4h=3 & h=6",
         {"atr": 4, "cl_4h": 3, "h": 6}),                                                     # #4
        ("atr=4 & cl_1h=0 & cl_4h=3 & squize_index=0",
         {"atr": 4, "cl_1h": 0, "cl_4h": 3, "squize_index": 0}),                              # #45
        ("atr=4 & cl_4h=3 & hill=2 & squize_index=0",
         {"atr": 4, "cl_4h": 3, "hill": 2, "squize_index": 0}),                               # #38
        ("atr=4 & cl_4h=3 & hill=2",
         {"atr": 4, "cl_4h": 3, "hill": 2}),                                                  # #10
        ("atr=4 & cl_4h=3 & dvol_minus_rv_12h=0",
         {"atr": 4, "cl_4h": 3, "dvol_minus_rv_12h": 0}),                                     # #37
        ("atr=4 & cl_4h=3 & cls_30m=5",
         {"atr": 4, "cl_4h": 3, "cls_30m": 5}),                                               # #24
        ("atr=4 & cl_4h=3 & squize_index=0",
         {"atr": 4, "cl_4h": 3, "squize_index": 0}),                                          # #43
        ("cl_1h=2 & cl_4h=3 & iv_est=4",
         {"cl_1h": 2, "cl_4h": 3, "iv_est": 4}),                                              # #39
        ("cl_4h=3 & h=6 & iv_est=4 & squize_index=0",
         {"cl_4h": 3, "h": 6, "iv_est": 4, "squize_index": 0}),                               # #27
        ("cl_4h=3 & h=6 & iv_est=4",
         {"cl_4h": 3, "h": 6, "iv_est": 4}),                                                  # #49
        ("cl_4h=3 & h=6 & hill=2",
         {"cl_4h": 3, "h": 6, "hill": 2}),                                                    # #48
        ("cl_4h=3 & h=6 & ret_6h=0",
         {"cl_4h": 3, "h": 6, "ret_6h": 0}),                                                  # #29
        ("cl_4h=3 & hill=2 & squize_index=0",
         {"cl_4h": 3, "hill": 2, "squize_index": 0}),                                         # #33
        ("cl_4h=3 & hill=2",
         {"cl_4h": 3, "hill": 2}),                                                            # #28
        ("cl_1h=0 & cl_4h=3 & squize_index=0",
         {"cl_1h": 0, "cl_4h": 3, "squize_index": 0}),                                        # #12
    ]
    cl_6 = _eval_group(cl_6_rules)

    # cl_7: семейство cls_30m=5 и super_cls=6
    cl_7_rules = [
        ("atr=4 & cls_30m=5 & iv_est=4", {"atr": 4, "cls_30m": 5, "iv_est": 4}),               # #11
        ("atr=4 & cls_30m=5 & squize_index=0", {"atr": 4, "cls_30m": 5, "squize_index": 0}),   # #34
        ("atr=4 & cls_30m=5", {"atr": 4, "cls_30m": 5}),                                       # #3
        ("cls_30m=5 & h=6 & squize_index=0", {"cls_30m": 5, "h": 6, "squize_index": 0}),       # #26
        ("cls_30m=5 & hill=0 & squize_index=0", {"cls_30m": 5, "hill": 0, "squize_index": 0}), # #44
        ("cls_30m=5 & squize_index=0", {"cls_30m": 5, "squize_index": 0}),                     # #1
        ("cls_15m=1 & super_cls=6", {"cls_15m": 1, "super_cls": 6}),                           # #31
        ("super_cls=6", {"super_cls": 6}),                                                     # #30
    ]
    cl_7 = _eval_group(cl_7_rules)

    # cl_8: семейство atr=4 & d=3 (без привязки к cl_4h=3)
    cl_8_rules = [
        ("atr=4 & d=3 & iv_est=4 & squize_index=0", {"atr": 4, "d": 3, "iv_est": 4, "squize_index": 0}),  # #13
        ("atr=4 & d=3 & squize_index=0", {"atr": 4, "d": 3, "squize_index": 0}),                           # #2
        ("atr=4 & d=3 & iv_est=4", {"atr": 4, "d": 3, "iv_est": 4}),                                       # #41
        ("atr=4 & d=3", {"atr": 4, "d": 3}),                                                               # #7
    ]
    cl_8 = _eval_group(cl_8_rules)

    # cl_9: семейство atr_ratio_24h_7d=4
    cl_9_rules = [
        ("atr_ratio_24h_7d=4 & cl_1d=4 & iv_est=4", {"atr_ratio_24h_7d": 4, "cl_1d": 4, "iv_est": 4}),    # #16
        ("atr_ratio_24h_7d=4 & cl_1d=4 & squize_index=0", {"atr_ratio_24h_7d": 4, "cl_1d": 4, "squize_index": 0}),  # #32
        ("atr_ratio_24h_7d=4 & cl_1h=3 & squize_index=0", {"atr_ratio_24h_7d": 4, "cl_1h": 3, "squize_index": 0}),  # #23
        ("atr_ratio_24h_7d=4 & cl_1h=3", {"atr_ratio_24h_7d": 4, "cl_1h": 3}),                             # #19
        ("atr_ratio_24h_7d=4", {"atr_ratio_24h_7d": 4}),                                                   # #47
    ]
    cl_9 = _eval_group(cl_9_rules)

    # cl_10: прочее (остаток второй таблицы)
    cl_10_rules = [
        ("cl_1h=0 & hill=0 & iv_est=4 & squize_index=0", {"cl_1h": 0, "hill": 0, "iv_est": 4, "squize_index": 0}),  # #18
        ("cl_1h=0 & hill=0 & iv_est=4", {"cl_1h": 0, "hill": 0, "iv_est": 4}),                                      # #8
        ("cl_1d=4 & hill=0 & iv_est=4", {"cl_1d": 4, "hill": 0, "iv_est": 4}),                                      # #9
        ("atr=4 & cl_1d=4 & cl_1h=0 & squize_index=0", {"atr": 4, "cl_1d": 4, "cl_1h": 0, "squize_index": 0}),      # #36
        ("atr=4 & hill=0 & iv_est=4 & squize_index=0", {"atr": 4, "hill": 0, "iv_est": 4, "squize_index": 0}),       # #46
        ("atr=4 & hill=0 & iv_est=4", {"atr": 4, "hill": 0, "iv_est": 4}),                                            # #35
        ("atr=4 & hill=0 & squize_index=0", {"atr": 4, "hill": 0, "squize_index": 0}),                                # #20
        ("atr=4 & dvol_minus_rv_12h=0 & hill=0", {"atr": 4, "dvol_minus_rv_12h": 0, "hill": 0}),                      # #42
        ("dvol_minus_rv_12h=0 & hill=0 & squize_index=0", {"dvol_minus_rv_12h": 0, "hill": 0, "squize_index": 0}),    # #14
        ("dvol_minus_rv_12h=0 & hill=0", {"dvol_minus_rv_12h": 0, "hill": 0}),                                         # #21
        ("atr=4 & cls_15m=1 & squize_index=0", {"atr": 4, "cls_15m": 1, "squize_index": 0}),                           # #22
        ("atr=4 & cl_4h=0", {"atr": 4, "cl_4h": 0}),                                                                    # #15
        ("atr=4 & cl_1h=2", {"atr": 4, "cl_1h": 2}),                                                                    # #6
        ("atr=4 & hill=0 & ret_6h=4", {"atr": 4, "hill": 0, "ret_6h": 4}),                                              # #40
        ("cl_15m=3 & cl_4h=4", {"cl_15m": 3, "cl_4h": 4}),                                                               # #17
    ]
    cl_10 = _eval_group(cl_10_rules)

    return {
        "cl_1": cl_1,
        "cl_2": cl_2,
        "cl_3": cl_3,
        "cl_4": cl_4,
        "cl_5": cl_5,
        "cl_6": cl_6,
        "cl_7": cl_7,
        "cl_8": cl_8,
        "cl_9": cl_9,
        "cl_10": cl_10,
    }


def get_comb_classes_2(x: dict) -> dict:
    """
    Кодирует набор “ban-паттернов” (из двух таблиц ниже) в cl_1..cl_10.

    - Вход: один словарь с фичами (можно передать trade-словарь; тогда возьмём x["features"], если он есть).
    - Выход: {"cl_1": int, ..., "cl_10": int}

    Внутри каждого cl_* правила идут сверху вниз по приоритету (сначала более специфичные).
    Возвращается ПЕРВОЕ совпавшее правило => внутри одного cl_* выбирается максимум одно значение.
    """

    # если передали trade-словарь, а не голые фичи
    if isinstance(x.get("features"), dict):
        f = x["features"]
    else:
        f = x

    def _get_int(key: str):
        v = f.get(key, None)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return v

    def _match(rule: dict) -> bool:
        for k, need in rule.items():
            if _get_int(k) != need:
                return False
        return True

    def _eval_group(rules: list) -> int:
        """
        rules: list[tuple[str, dict]]
        Возвращает:
          0 если ничего не совпало, иначе 1..len(rules) по порядку.
        """
        for i, (_, cond) in enumerate(rules, 1):
            if _match(cond):
                return i
        return 0

    # =========================================================================
    # FIRST TABLE (1..50) -> распределено по cl_1..cl_5
    # =========================================================================

    # cl_1: семейство cl_1d=2 & cl_4h=2
    cl_1_rules = [
        ("cl_1d=2 & cl_4h=2 & hill=0", {"cl_1d": 2, "cl_4h": 2, "hill": 0}),                  # #11
        ("cl_1d=2 & cl_4h=2 & h=12", {"cl_1d": 2, "cl_4h": 2, "h": 12}),                       # #34
        ("cl_1d=2 & cl_4h=2 & squize_index=0", {"cl_1d": 2, "cl_4h": 2, "squize_index": 0}),   # #27
        ("cl_1d=2 & cl_4h=2", {"cl_1d": 2, "cl_4h": 2}),                                       # #2
    ]
    cl_1 = _eval_group(cl_1_rules)

    # cl_2: семейство d=3 (и вариации вокруг d=3)
    cl_2_rules = [
        ("d=3 & rsi=4 & squize_index=0", {"d": 3, "rsi": 4, "squize_index": 0}),               # #5
        ("cl_1h=3 & d=3 & squize_index=0", {"cl_1h": 3, "d": 3, "squize_index": 0}),           # #8
        ("cl_1h=3 & d=3", {"cl_1h": 3, "d": 3}),                                               # #1
        ("atr_ratio_24h_7d=4 & d=3 & squize_index=0", {"atr_ratio_24h_7d": 4, "d": 3, "squize_index": 0}),  # #18
        ("d=3 & hill=0 & squize_index=0", {"d": 3, "hill": 0, "squize_index": 0}),             # #9
        ("d=3 & rsi=4", {"d": 3, "rsi": 4}),                                                   # #7
        ("d=3 & h=6 & squize_index=0", {"d": 3, "h": 6, "squize_index": 0}),                   # #35
        ("atr=2 & d=3", {"atr": 2, "d": 3}),                                                   # #20
        ("atr_ratio_24h_7d=4 & d=3", {"atr_ratio_24h_7d": 4, "d": 3}),                         # #38
        ("d=3 & hill=0", {"d": 3, "hill": 0}),                                                 # #10
        ("d=3 & squize_index=0", {"d": 3, "squize_index": 0}),                                 # #3
        ("cl_1d=3 & d=3 & squize_index=0", {"cl_1d": 3, "d": 3, "squize_index": 0}),           # #44
        ("d=3", {"d": 3}),                                                                     # #4
    ]
    cl_2 = _eval_group(cl_2_rules)

    # cl_3: семейство cl_1h=3 + cl_15m=2 + cl_4h=1 (то, что крутится вокруг этих)
    cl_3_rules = [
        ("atr=2 & cl_1h=3 & dvol_minus_rv_12h=4 & hill=0",
         {"atr": 2, "cl_1h": 3, "dvol_minus_rv_12h": 4, "hill": 0}),                           # #49

        ("cl_15m=2 & cl_1h=3 & hill=0 & squize_index=0",
         {"cl_15m": 2, "cl_1h": 3, "hill": 0, "squize_index": 0}),                             # #28
        ("cl_15m=2 & cl_1h=3 & squize_index=0",
         {"cl_15m": 2, "cl_1h": 3, "squize_index": 0}),                                        # #16
        ("cl_15m=2 & cl_1h=3 & hill=0",
         {"cl_15m": 2, "cl_1h": 3, "hill": 0}),                                                # #40
        ("cl_15m=2 & cl_1h=3",
         {"cl_15m": 2, "cl_1h": 3}),                                                           # #15

        ("cl_1d=3 & cl_1h=3 & hill=0",
         {"cl_1d": 3, "cl_1h": 3, "hill": 0}),                                                 # #17

        ("cl_1h=3 & cls_30m=3 & hill=0",
         {"cl_1h": 3, "cls_30m": 3, "hill": 0}),                                               # #29
        ("cl_1h=3 & cls_30m=3",
         {"cl_1h": 3, "cls_30m": 3}),                                                          # #37

        ("cl_1h=3 & cl_4h=1 & squize_index=0",
         {"cl_1h": 3, "cl_4h": 1, "squize_index": 0}),                                         # #48
        ("cl_1h=3 & cl_4h=1",
         {"cl_1h": 3, "cl_4h": 1}),                                                            # #50

        ("atr_ratio_24h_7d=0 & cl_1h=3",
         {"atr_ratio_24h_7d": 0, "cl_1h": 3}),                                                 # #41
    ]
    cl_3 = _eval_group(cl_3_rules)

    # cl_4: семейство cl_1d=2 & iv_est=2 и рядом (d=6 блок, iv_est=2 блок)
    cl_4_rules = [
        ("atr_ratio_24h_7d=0 & cl_1d=2 & d=6 & hill=0",
         {"atr_ratio_24h_7d": 0, "cl_1d": 2, "d": 6, "hill": 0}),                              # #36
        ("atr_ratio_24h_7d=0 & cl_1d=2 & d=6",
         {"atr_ratio_24h_7d": 0, "cl_1d": 2, "d": 6}),                                         # #12

        ("cl_1d=2 & hill=0 & iv_est=2",
         {"cl_1d": 2, "hill": 0, "iv_est": 2}),                                                # #14
        ("cl_4h=2 & hill=0 & iv_est=2",
         {"cl_4h": 2, "hill": 0, "iv_est": 2}),                                                # #21

        ("cl_1d=2 & iv_est=2 & squize_index=0",
         {"cl_1d": 2, "iv_est": 2, "squize_index": 0}),                                        # #42
        ("cl_1d=2 & h=6 & iv_est=2",
         {"cl_1d": 2, "h": 6, "iv_est": 2}),                                                   # #43
        ("cl_1d=2 & iv_est=2",
         {"cl_1d": 2, "iv_est": 2}),                                                           # #6

        ("dvol_minus_rv_12h=3 & iv_est=2",
         {"dvol_minus_rv_12h": 3, "iv_est": 2}),                                               # #45
    ]
    cl_4 = _eval_group(cl_4_rules)

    # cl_5: прочее из первой таблицы (cls_1h=6, dvol/h12, остатки cl_4h=2, и т.д.)
    cl_5_rules = [
        ("cl_1d=2 & dvol_minus_rv_12h=3 & h=12",
         {"cl_1d": 2, "dvol_minus_rv_12h": 3, "h": 12}),                                       # #31

        ("dvol_minus_rv_12h=3 & h=12 & hill=0",
         {"dvol_minus_rv_12h": 3, "h": 12, "hill": 0}),                                        # #30
        ("dvol_minus_rv_12h=3 & h=12 & squize_index=0",
         {"dvol_minus_rv_12h": 3, "h": 12, "squize_index": 0}),                                # #39

        ("cls_1h=6 & h=12 & hill=0",
         {"cls_1h": 6, "h": 12, "hill": 0}),                                                   # #19
        ("cl_1d=2 & cls_1h=6 & h=12",
         {"cl_1d": 2, "cls_1h": 6, "h": 12}),                                                  # #25
        ("cl_1d=2 & cls_1h=6",
         {"cl_1d": 2, "cls_1h": 6}),                                                           # #13
        ("cls_1h=6 & hill=0",
         {"cls_1h": 6, "hill": 0}),                                                            # #24

        ("cl_1d=2 & cls_30m=4",
         {"cl_1d": 2, "cls_30m": 4}),                                                          # #22

        ("cl_4h=2 & dvol_minus_rv_12h=3 & hill=0",
         {"cl_4h": 2, "dvol_minus_rv_12h": 3, "hill": 0}),                                     # #47
        ("cl_4h=2 & dvol_minus_rv_12h=3",
         {"cl_4h": 2, "dvol_minus_rv_12h": 3}),                                                # #46

        ("cl_15m=0 & cl_4h=2 & hill=0",
         {"cl_15m": 0, "cl_4h": 2, "hill": 0}),                                                # #33
        ("cl_15m=1 & cl_4h=2",
         {"cl_15m": 1, "cl_4h": 2}),                                                           # #49

        ("cl_4h=2 & hill=0 & rsi=1",
         {"cl_4h": 2, "hill": 0, "rsi": 1}),                                                   # #23

        ("cl_4h=2 & squize_index=4",
         {"cl_4h": 2, "squize_index": 4}),                                                     # #32
    ]
    cl_5 = _eval_group(cl_5_rules)

    # =========================================================================
    # SECOND TABLE (1..50) -> распределено по cl_6..cl_10
    # =========================================================================

    # cl_6: семейство h=12
    cl_6_rules = [
        ("h=12 & squize_index=0", {"h": 12, "squize_index": 0}),                                # #5
        ("dvol_minus_rv_12h=4 & h=12", {"dvol_minus_rv_12h": 4, "h": 12}),                      # #37
        ("atr_ratio_24h_7d=1 & h=12", {"atr_ratio_24h_7d": 1, "h": 12}),                        # #27
        ("cl_1h=2 & h=12", {"cl_1h": 2, "h": 12}),                                              # #34
        ("d=2 & h=12", {"d": 2, "h": 12}),                                                      # #30
        ("cl_1d=0 & h=12", {"cl_1d": 0, "h": 12}),                                              # #35
        ("h=12", {"h": 12}),                                                                    # #1
    ]
    cl_6 = _eval_group(cl_6_rules)

    # cl_7: семейство cl_1d=0 (кроме случаев h=12, которые уже в cl_6)
    cl_7_rules = [
        ("cl_1d=0 & cl_1h=2", {"cl_1d": 0, "cl_1h": 2}),                                        # #12
        ("cl_1d=0 & squize_index=0", {"cl_1d": 0, "squize_index": 0}),                          # #7
        ("cl_1d=0", {"cl_1d": 0}),                                                              # #3
    ]
    cl_7 = _eval_group(cl_7_rules)

    # cl_8: семейство d=2 / d=6
    cl_8_rules = [
        ("d=2 & hill=0", {"d": 2, "hill": 0}),                                                  # #14
        ("d=2 & squize_index=0", {"d": 2, "squize_index": 0}),                                  # #21
        ("d=2", {"d": 2}),                                                                      # #2
        ("d=6 & squize_index=0", {"d": 6, "squize_index": 0}),                                  # #43
        ("d=6", {"d": 6}),                                                                      # #20
    ]
    cl_8 = _eval_group(cl_8_rules)

    # cl_9: семейство atr=4 / atr_ratio_24h_7d / atr=3 & dvol=4
    cl_9_rules = [
        ("atr=4 & dvol_minus_rv_12h=0 & squize_index=0", {"atr": 4, "dvol_minus_rv_12h": 0, "squize_index": 0}),  # #18
        ("atr=4 & dvol_minus_rv_12h=0", {"atr": 4, "dvol_minus_rv_12h": 0}),                                      # #13
        ("atr=4 & hill=2 & squize_index=0", {"atr": 4, "hill": 2, "squize_index": 0}),                            # #29
        ("atr=4 & hill=2", {"atr": 4, "hill": 2}),                                                                # #26
        ("atr=4 & squize_index=0", {"atr": 4, "squize_index": 0}),                                                # #45
        ("atr=4", {"atr": 4}),                                                                                    # #11
        ("atr_ratio_24h_7d=1", {"atr_ratio_24h_7d": 1}),                                                          # #10
        ("atr_ratio_24h_7d=3", {"atr_ratio_24h_7d": 3}),                                                          # #46
        ("atr=3 & dvol_minus_rv_12h=4", {"atr": 3, "dvol_minus_rv_12h": 4}),                                      # #47
    ]
    cl_9 = _eval_group(cl_9_rules)

    # cl_10: остаток второй таблицы
    cl_10_rules = [
        ("hill=2 & iv_est=4 & ret_6h=0 & squize_index=0", {"hill": 2, "iv_est": 4, "ret_6h": 0, "squize_index": 0}),  # #44
        ("iv_est=4 & ret_6h=0 & squize_index=0", {"iv_est": 4, "ret_6h": 0, "squize_index": 0}),                      # #36
        ("hill=2 & iv_est=4 & squize_index=0", {"hill": 2, "iv_est": 4, "squize_index": 0}),                          # #39
        ("cl_4h=0 & squize_index=0", {"cl_4h": 0, "squize_index": 0}),                                                # #19
        ("cl_4h=3 & hill=0", {"cl_4h": 3, "hill": 0}),                                                                # #25
        ("cl_15m=2 & iv_est=3", {"cl_15m": 2, "iv_est": 3}),                                                          # #32

        ("hill=2 & squize_index=0", {"hill": 2, "squize_index": 0}),                                                  # #15
        ("dvol_minus_rv_12h=2 & squize_index=0", {"dvol_minus_rv_12h": 2, "squize_index": 0}),                        # #38
        ("iv_est=4 & ret_6h=0", {"iv_est": 4, "ret_6h": 0}),                                                          # #41
        ("cl_1h=2 & hill=0", {"cl_1h": 2, "hill": 0}),                                                                # #42
        ("cl_1d=3 & iv_est=2 & squize_index=0", {"cl_1d": 3, "iv_est": 2, "squize_index": 0}),                        # #48
        ("hill=1 & ret_6h=4", {"hill": 1, "ret_6h": 4}),                                                              # #50

        ("cls_30m=7 & hill=0", {"cls_30m": 7, "hill": 0}),                                                            # #33
        ("cl_15m=2", {"cl_15m": 2}),                                                                                  # #23
        ("cls_30m=7", {"cls_30m": 7}),                                                                                 # #24
        ("cls_1h=7", {"cls_1h": 7}),                                                                                   # #28
        ("dvol_minus_rv_12h=2", {"dvol_minus_rv_12h": 2}),                                                             # #22
        ("ret_6h=0", {"ret_6h": 0}),                                                                                   # #40
        ("iv_est=4", {"iv_est": 4}),                                                                                   # #17
        ("hill=2", {"hill": 2}),                                                                                       # #8
        ("cl_4h=3", {"cl_4h": 3}),                                                                                      # #9
        ("cl_4h=0", {"cl_4h": 0}),                                                                                      # #6
        ("cls_30m=5", {"cls_30m": 5}),                                                                                  # #16
        ("cl_1h=2", {"cl_1h": 2}),                                                                                      # #4
        ("cl_1h=1", {"cl_1h": 1}),                                                                                      # #31
    ]
    cl_10 = _eval_group(cl_10_rules)

    return {
        "cl_1": cl_1,
        "cl_2": cl_2,
        "cl_3": cl_3,
        "cl_4": cl_4,
        "cl_5": cl_5,
        "cl_6": cl_6,
        "cl_7": cl_7,
        "cl_8": cl_8,
        "cl_9": cl_9,
        "cl_10": cl_10,
    }
