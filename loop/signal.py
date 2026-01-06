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
import time
import loop.risk as risk
from loop.strategy_1 import get_best_combination
from loop.risk import RiskRegimeResult
from loop.two_stage_prod_infer import predict_one
from loop.get_comb_classes import get_comb_classes_1, get_comb_classes_0, get_comb_classes_2
from loop.quant_to_cl import quantile_classes_0_4
from markov.hmm import train_hmm_regime_model, FeatureConfig, load_hmm_regime_model, infer_regime_latest,current_regime_duration
from loop.trade_pass_filter import calc_long_short, calc_long_short_v2, calc_long_short_v3, calc_long_short_v4
import talib





def get_signal(i, start):
    sv.dt = datetime.fromtimestamp(sv.data_1h[i][0]/1000)
    dow = sv.dt.weekday()
    sv.timestamp = sv.data_1h[i][0]
    sv.hour = sv.dt.hour
    sv.dow = dow
    
    if dow in [] or sv.hour not in [start]:
        return 0
    
    pic = sv.precalc_pic.get(sv.data_1h[i][0], None)
    
    if pic is None:
        return 0
    
    
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
    
    
    sv.iv_est = get_iv_26h(sv.data_1h[i-1][0], csv_path='/home/jupiter/PYTHON/MARKET_DATA/BTC_DVOL_3600s_20210101_20260106.csv')
    if sv.iv_est is None:
        return 0
    
    sv.hill = 1 if sv.data_1h[i][1] * (1-0.01) > sv.data_1h[i-9][1] else 2 if sv.data_1h[i][1] * (1+0.01) < sv.data_1h[i-9][1] else 0
    atr_raw = talib.ATR(sv.data_1h[i-60:i, 2], sv.data_1h[i-60:i, 3], sv.data_1h[i-60:i, 4], timeperiod=14)[-1]
    sv.risk_usd = atr_raw * (sv.amount/sv.data_1h[i][1])
    sv.atr = atr_raw/sv.data_1h[i][1]
    sv.rsi = talib.RSI(sv.data_1h[i-60:i, 4], timeperiod=14)[-1]
    sv.squeeze_index, sv.squeeze_count = plot_kc_bb_squeeze_np(sv.data_1h[i - 84:i], "test", save_image=False)
    d_fut = quantile_classes_0_4(rsi=sv.rsi, atr=sv.atr, iv_est=sv.iv_est, squize_index=sv.squeeze_index)
    
    # day_start = util.day_start_ts_ms_utc(int(sv.data_1h[i][0]))
    # day_index = find_candle_index(day_start, sv.data_1d)

    # if day_index < 180:
    #     return 0

    # # ВАЖНО: добавляем +1, чтобы текущий день тоже попал в окно
    # days_cand = sv.data_1d[day_index - 180 : day_index]
    # h6 = max(sv.data_1h[i-6:i, 2])
    # l6 = max(sv.data_1h[i-6:i, 3])
    # last_day_6h_np = np.asarray([
    #     day_start,
    #     sv.data_1h[i-6, 1],
    #     h6,
    #     l6,
    #     sv.data_1h[i-1, 4],
    #     np.sum(sv.data_1h[i-6:i, 5]),
    # ], dtype=np.float64).reshape(1, 6)
    # full_days_cand = np.vstack([days_cand, last_day_6h_np])
    # res = infer_regime_latest(
    #     full_days_cand,
    #     sv.regime_model_1,
    #     lookback_days=180,
    #     return_history=True,
    #     causal_probs=False,
    # )

    # state_like_csv = int(res["states"][-1])   # Viterbi

    

    # sv.reg_d = state_like_csv# res['current_state']
    
    # res_2 = infer_regime_latest(
    #     sv.data_1h[i-180:i],
    #     sv.regime_model_2,
    #     lookback_days=180,
    #     return_history=True,
    #     causal_probs=False,
    # )
    
    # sv.reg_h = int(res_2["states"][-1]) 
    # # util.print_regimes_colored(sv.reg_d, state_like_csv, sv.reg_h)
    # dur_d = current_regime_duration(res, use_viterbi=True)
    # dur_h = current_regime_duration(res_2, use_viterbi=True)
    # sv.dur_d = dur_d['bars']
    # sv.dur_h = dur_h['bars']
    
    
    # #================CHECK===================
    # ch_1 = infer_regime_latest(
    #     full_days_cand,
    #     sv.regime_model_3,
    #     lookback_days=180,
    #     return_history=True,
    #     causal_probs=False,
    # )
    
    # sv.ch_1d = int(ch_1["states"][-1])

    vars_1 = {
        'dow': dow,
        'atr': d_fut['atr'],
        'iv_est': sv.iv_est,
        'rsi': sv.rsi,
        'hill': sv.hill,
        'cl_4h': sv.cl_4h
    }
    
    rules_1 = {
        'dow': [0,1,2,4,6],
        'atr': [0,3,4],
        'iv_est': '>0.33',
        'rsi': '>31'
    }
    
    rules_2 = {
        'dow': [1,3,4],
        'atr': [1,2],
        'hill': [0,2],
        'cl_4h': [1,2,3,4],
        'rsi': '<76',
    }
    
    long, short, exp = calc_long_short_v4(vars_1=vars_1, rules_1=rules_1, rules_2=rules_2)

    if short:
        return 2
    if long:
        return 1

    return 0

 