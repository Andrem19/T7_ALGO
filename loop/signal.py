from loop import get_comb_classes
import shared_vars as sv
from datetime import datetime
import pandas as pd
from loop.btc_dvol_iv import get_iv_26h
from loop.dvol_features_6am import calc_features_at_ts
from loop.squeeze import plot_kc_bb_squeeze_np
from helpers.util import find_candle_index
import helpers.util as util
import numpy as np
import loop.risk as risk
from loop.risk import RiskRegimeResult
from loop.two_stage_prod_infer import predict_one
from loop.get_comb_classes import get_comb_classes_1, get_comb_classes_0, get_comb_classes_2
from loop.quant_to_cl import quantile_classes_0_4
from markov.hmm import train_hmm_regime_model, FeatureConfig, load_hmm_regime_model, infer_regime_latest,current_regime_duration
from loop.trade_pass_filter import calc_long_short
import talib





def get_signal(i, start):
    dt = datetime.fromtimestamp(sv.data_1h[i][0]/1000)
    dow = dt.weekday()
    sv.hour = dt.hour
    sv.dow = dow
    
    if dow in [] or sv.hour not in [start]:
        return 0
    
    pic = sv.precalc_pic.get(sv.data_1h[i][0], None)
    
    if pic is None:
        return 0
    
    # sv.feats = calc_features_at_ts(sv.data_1h[i][0], sv.ctx)
    
    
    sv.cl_1d = pic['cl_1d']
    sv.cl_4h = pic['cl_4h']
    sv.cl_1h = pic['cl_1h']
    sv.cl_15m = pic['cl_15m']
    
    cls_dict = sv.cls_t.get(sv.data_1h[i][0], None)
    
    if cls_dict is None:
        return 0
    
    sv.cls_1h = cls_dict['1h']
    sv.cls_30m = cls_dict['30m']
    sv.cls_15m = cls_dict['15m']
    sv.cls_5m = cls_dict['5m']
    sv.super_cls = cls_dict['super_cls']
    
    
    sv.iv_est = get_iv_26h(sv.data_1h[i-1][0], csv_path='BTC_DVOL_3600s_20200101_20251212.csv')
    if sv.iv_est is None:
        return 0
    
    sv.hill = 1 if sv.data_1h[i][1] * (1-0.01) > sv.data_1h[i-9][1] else 2 if sv.data_1h[i][1] * (1+0.01) < sv.data_1h[i-9][1] else 0
    atr_raw = talib.ATR(sv.data_1h[i-60:i, 2], sv.data_1h[i-60:i, 3], sv.data_1h[i-60:i, 4], timeperiod=14)[-1]
    sv.risk_usd = atr_raw * (sv.amount/sv.data_1h[i][1])
    sv.atr = atr_raw/sv.data_1h[i][1]
    sv.rsi = talib.RSI(sv.data_1h[i-60:i, 4], timeperiod=14)[-1]
    sv.squeeze_index, sv.squeeze_count = plot_kc_bb_squeeze_np(sv.data_1h[i - 84:i], "test", save_image=False)
    
    
    # state_1 = sv.regime_state_1.get(util.day_start_ts_ms_utc(int(sv.data_1h[i][0])), None)
    # if state_1 is None:
    #     return 0
    # state_2 = sv.regime_state_2.get(sv.data_1h[i][0], None)
    # if state_2 is None:
    #     return 0


    # sv.reg_d = state_1["state"]
    # sv.reg_h = state_2["state"]

    # sv.super_r = sv.reg_d * 6 + sv.reg_h
    
    
    # sv.futures_1 = {
    #         'hill': sv.hill,
    #         'cls_1h': int(sv.cls_1h),
    #         'cls_30m': int(sv.cls_30m),
    #         'cls_15m': int(sv.cls_15m),
    #         'cls_5m': int(sv.cls_5m),
    #         'super_cls': int(sv.super_cls),
    #         'd': sv.dow,
    #         'h': sv.hour,
    #         'cl_1d': int(sv.cl_1d),
    #         'cl_4h': int(sv.cl_4h),
    #         'cl_1h': int(sv.cl_1h),
    #         'cl_15m': int(sv.cl_15m),
    # }
    
    # sv.futures_1.update(quantile_classes_0_4(rsi=sv.rsi, atr=sv.atr, iv_est=sv.iv_est, squize_index=sv.squeeze_count, ret_6h=sv.feats['ret_6h'], dvol_minus_rv_12h=sv.feats['dvol_minus_rv_12h'], atr_ratio_24h_7d=sv.feats['atr_ratio_24h_7d']))
    

    # if state_1['state'] == 0:
    #     sv.comb_classes = get_comb_classes_0(sv.futures_1)
    # elif state_1['state'] == 1:
    #     sv.comb_classes = get_comb_classes_1(sv.futures_1)
    # elif state_1['state'] == 2:
    #     sv.comb_classes = get_comb_classes_2(sv.futures_1)
    day_start = util.day_start_ts_ms_utc(int(sv.data_1h[i][0]))
    day_index = find_candle_index(day_start, sv.data_1d)

    if day_index < 180:
        return 0

    # ВАЖНО: добавляем +1, чтобы текущий день тоже попал в окно
    days_cand = sv.data_1d[day_index - 180 : day_index]
    h6 = max(sv.data_1h[i-6:i, 2])
    l6 = max(sv.data_1h[i-6:i, 3])
    last_day_6h_np = np.asarray([
        day_start,
        sv.data_1h[i-6, 1],
        h6,
        l6,
        sv.data_1h[i-1, 4],
        np.sum(sv.data_1h[i-6:i, 5]),
    ], dtype=np.float64).reshape(1, 6)
    full_days_cand = np.vstack([days_cand, last_day_6h_np])
    res = infer_regime_latest(
        full_days_cand,
        sv.regime_model_1,
        lookback_days=180,
        return_history=True,
        causal_probs=False,
    )

    state_like_csv = int(res["states"][-1])   # Viterbi

    

    sv.reg_d = state_like_csv# res['current_state']
    
    res_2 = infer_regime_latest(
        sv.data_1h[i-180:i],
        sv.regime_model_2,
        lookback_days=180,
        return_history=True,
        causal_probs=False,
    )
    
    sv.reg_h = int(res_2["states"][-1]) 
    # util.print_regimes_colored(sv.reg_d, state_like_csv, sv.reg_h)
    dur_d = current_regime_duration(res, use_viterbi=True)
    dur_h = current_regime_duration(res_2, use_viterbi=True)
    sv.dur_d = dur_d['bars']
    sv.dur_h = dur_h['bars']
    
    
    #================CHECK===================
    ch_1 = infer_regime_latest(
        full_days_cand,
        sv.regime_model_3,
        lookback_days=180,
        return_history=True,
        causal_probs=False,
    )
    
    ch_1d = int(ch_1["states"][-1])



    # long = dow not in [3,5] and sv.reg_d not in [1] and sv.reg_d == ch_1d and not (dow in [4] and sv.reg_d in [0])
    # short = dow not in [2,5,6] and sv.reg_d not in [2] and sv.reg_d == ch_1d and sv.reg_h not in [0,4,5]

    # long_2 = ((ch_1d in [2] and dow not in [3]) or (dow in [2] and ch_1d not in [1])) and dow not in [5] and sv.reg_h not in [4]
    # short_2 = ((ch_1d in [1] and dow not in [2]) or dow in [3]) and dow not in [5,6] and sv.reg_h not in [0]


    # if long and long == long_2:
    #     return 1
    # if short and short == short_2:
    #     return 2   

    # return 0
    
    # reg_d = sv.reg_d
    # reg_h = sv.reg_h
    # same = (reg_d == ch_1d)

    # long = (
    #     same
    #     and ch_1d != 1
    #     and dow not in {3, 5}
    #     and not (dow == 4 and ch_1d == 0)
    #     and dow != 5
    #     and reg_h != 4
    #     and ((ch_1d == 2 and dow != 3) or (dow == 2 and ch_1d != 1))
    # )

    # short = (
    #     same
    #     and ch_1d != 2
    #     and dow not in {2, 5, 6}
    #     and reg_h not in {0, 4, 5}
    #     and dow not in {5, 6}
    #     and reg_h != 0
    #     and ((ch_1d == 1 and dow != 2) or (dow == 3))
    # )
    long, short, exp = calc_long_short(dow=dow, ch_1d=ch_1d, reg_d=sv.reg_d, reg_h=sv.reg_h)

    if long:
        return 1
    if short:
        return 2
    return 0

 
