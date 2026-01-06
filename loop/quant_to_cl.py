from __future__ import annotations

from typing import Dict


def quantile_classes_0_4(
    *,
    rsi: float,
    atr: float,
    iv_est: float,
    squize_index: float,
) -> Dict[str, int]:
    """
    Принимает 7 фич и возвращает класс 0..4 для каждой по границам квантилей.

    Правило выхода за границы:
      - если значение ниже нижней границы (q0) -> класс 0
      - если значение выше верхней границы (q100) -> класс 4

    Интервалы:
      0: ниже q20
      1: от q20 (включая) до q40 (не включая)
      2: от q40 (включая) до q60 (не включая)
      3: от q60 (включая) до q80 (не включая)
      4: от q80 (включая) и выше
    """

    # ---------------------------
    # 1) Quantile bounds (hardcoded)
    # ---------------------------
    q_rsi_0 = 8.214468393363248
    q_rsi_20 = 41.340878269476875
    q_rsi_40 = 47.87883118386921
    q_rsi_60 = 53.0029671539982
    q_rsi_80 = 59.48455201886614
    q_rsi_100 = 87.5088827002338

    # q_atr_0 = 0.0010695035518906
    # q_atr_20 = 0.004
    # q_atr_40 = 0.00687444297853534
    # q_atr_60 = 0.012
    # q_atr_80 = 0.03
    # q_atr_100 = 0.0529649173754248
    
    q_atr_0 = 0.0010695035518906
    q_atr_20 = 0.00527083189877616
    q_atr_40 = 0.00687444297853534
    q_atr_60 = 0.0085267112865776
    q_atr_80 = 0.010959036282887387
    q_atr_100 = 0.0529649173754248

    q_iv_0 = 0.01
    q_iv_20 = 0.2412272805942709
    q_iv_40 = 0.4694442124789252
    q_iv_60 = 0.6250107028717983
    q_iv_80 = 0.8028167639972871
    q_iv_100 = 2.9041674622894766

    q_sq_0 = 0.2789805107746579
    q_sq_20 = 0.8132108668695694
    q_sq_40 = 1.0892525936004012
    q_sq_60 = 1.4300622374949454
    q_sq_80 = 2.004429370443966
    q_sq_100 = 5.935761184896639

    q_ret_0 = -11.36777214056554
    q_ret_20 = -0.7392249788642199
    q_ret_40 = -0.16378587085619
    q_ret_60 = 0.1865707116871673
    q_ret_80 = 0.6925869046447914
    q_ret_100 = 10.532445486100992

    q_dmr_0 = -155.75478723371248
    q_dmr_20 = 7.382851354631276
    q_dmr_40 = 21.327954317544002
    q_dmr_60 = 29.58019678714698
    q_dmr_80 = 39.30926352283976
    q_dmr_100 = 83.62837710429478

    q_ar_0 = 0.2709933911606742
    q_ar_20 = 0.7737811062502306
    q_ar_40 = 0.9125171449659456
    q_ar_60 = 1.031875130126236
    q_ar_80 = 1.2127188651441316
    q_ar_100 = 3.190825475669305

    # ---------------------------
    # 2) Classifier logic inline (no helper)
    # ---------------------------
    # RSI
    x = rsi
    if x < q_rsi_0:
        x = q_rsi_0
    elif x > q_rsi_100:
        x = q_rsi_100
    if x < q_rsi_20:
        cls_rsi = 0
    elif x < q_rsi_40:
        cls_rsi = 1
    elif x < q_rsi_60:
        cls_rsi = 2
    elif x < q_rsi_80:
        cls_rsi = 3
    else:
        cls_rsi = 4

    # ATR
    x = atr
    if x < q_atr_0:
        x = q_atr_0
    elif x > q_atr_100:
        x = q_atr_100
    if x < q_atr_20:
        cls_atr = 0
    elif x < q_atr_40:
        cls_atr = 1
    elif x < q_atr_60:
        cls_atr = 2
    elif x < q_atr_80:
        cls_atr = 3
    else:
        cls_atr = 4

    # IV_EST
    x = iv_est
    if x < q_iv_0:
        x = q_iv_0
    elif x > q_iv_100:
        x = q_iv_100
    if x < q_iv_20:
        cls_iv = 0
    elif x < q_iv_40:
        cls_iv = 1
    elif x < q_iv_60:
        cls_iv = 2
    elif x < q_iv_80:
        cls_iv = 3
    else:
        cls_iv = 4

    # SQUIZE_INDEX
    x = squize_index
    if x < q_sq_0:
        x = q_sq_0
    elif x > q_sq_100:
        x = q_sq_100
    if x < q_sq_20:
        cls_sq = 0
    elif x < q_sq_40:
        cls_sq = 1
    elif x < q_sq_60:
        cls_sq = 2
    elif x < q_sq_80:
        cls_sq = 3
    else:
        cls_sq = 4

    return {
        "rsi": cls_rsi,
        "atr": cls_atr,
        "iv_est": cls_iv,
        "squize_index": cls_sq
    }
