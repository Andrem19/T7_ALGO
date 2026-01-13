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
    
    if dow in []:# or sv.hour not in [start]:
        return 0
    
    
    vec = sv.vector.get(sv.data_1h[i][0], None)
    
    if vec is None:
        return 0
    
    # pic = sv.precalc_pic.get(sv.data_1h[i][0], None)
    
    # if pic is None:
    #     return 0
    
    
    sv.cl_1d = vec['cl_1d']
    sv.cl_4h = vec['cl_4h']
    sv.cl_1h = vec['cl_1h']
    sv.cl_15m = vec['cl_15m']
    
    # cls_dict = sv.cls_t.get(sv.data_1h[i][0], None)
    
    # if cls_dict is None:
    #     return 0
    
    sv.cls_1h = vec['cls_1h']
    sv.cls_30m = vec['cls_30m']
    sv.cls_15m = vec['cls_15m']
    sv.cls_5m = vec['cls_5m']
    sv.super_cls = vec['super_cls']
    
    
    sv.iv_est = vec['iv_est_1']

    
    sv.hill = vec['hill']#1 if sv.data_1h[i][1] * (1-0.01) > sv.data_1h[i-9][1] else 2 if sv.data_1h[i][1] * (1+0.01) < sv.data_1h[i-9][1] else 0
    atr_raw = talib.ATR(sv.data_1h[i-60:i, 2], sv.data_1h[i-60:i, 3], sv.data_1h[i-60:i, 4], timeperiod=14)[-1]
    sv.risk_usd = atr_raw * (sv.amount/sv.data_1h[i][1])
    sv.atr = atr_raw/sv.data_1h[i][1]
    sv.rsi = vec['rsi_1']#talib.RSI(sv.data_1h[i-60:i, 4], timeperiod=14)[-1]
    sv.squeeze_index = vec['squize_index_1']#, sv.squeeze_count = plot_kc_bb_squeeze_np(sv.data_1h[i - 84:i], "test", save_image=False)
    
    sv.vix = vec['vix']
    sv.sp500 = vec['sp500']
    sv.fg = vec['feer_and_greed']
    sv.fg_stock = vec['fg_stock']
    
    # day_start = util.day_start_ts_ms_utc(int(sv.data_1h[i][0]))
    # day_index = find_candle_index(day_start, sv.data_1d)

    # if day_index < 180:
    #     return 0

    # ВАЖНО: добавляем +1, чтобы текущий день тоже попал в окно
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

    sv.reg_d = vec['reg_d']# int(res["states"][-1])
    
    # res_2 = infer_regime_latest(
    #     sv.data_1h[i-180:i],
    #     sv.regime_model_2,
    #     lookback_days=180,
    #     return_history=True,
    #     causal_probs=False,
    # )
    
    sv.reg_h = vec['reg_h']#int(res_2["states"][-1]) 
    
    
    
    vars_1 = {
        'feer_and_greed': sv.fg,
        'fg_stock': sv.fg_stock,
        'sp500': sv.sp500,
        'vix': sv.vix,
        'rsi_1': float(sv.rsi),
        'atr_1': float(sv.atr),
        'iv_est_1': float(sv.iv_est),
        'rsi': vec['rsi'],
        'atr': vec['atr'],
        'iv_est': vec['iv_est'],
        'squize_index': vec['squize_index'],
        'squize_index_1': float(sv.squeeze_index),
        'hill': sv.hill,
        'reg_d': sv.reg_d,
        'reg_h': sv.reg_h,
        'd': sv.dow,
        'h': sv.hour,
        'cls_1h': int(sv.cls_1h),
        'cls_30m': int(sv.cls_30m),
        'cls_15m': int(sv.cls_15m),
        'cls_5m': int(sv.cls_5m),
        'super_cls': int(sv.super_cls),
        'cl_1d': int(sv.cl_1d),
        'cl_4h': int(sv.cl_4h),
        'cl_1h': int(sv.cl_1h),
        'cl_15m': int(sv.cl_15m),
    }
    
    #long_candels = (int((sv.data_1h[i-12:i, 4] > sv.data_1h[i-12:i, 1]).sum()) - int((sv.data_1h[i-12:i, 4] < sv.data_1h[i-12:i, 1]).sum()))


    short =  sv.dow in [3] and sum([sv.cl_1d in [2], sv.cl_15m in [3], sv.cl_1h in [1], sv.hour in [15]])>=2
    # if not short:
    #     short = sv.dow in [3] and (sv.cl_1d==2 or (sv.cl_15m==3 and sv.cl_1h==1))
    
    long = sv.dow in [0,1,2,3,6] and sum([vec['iv_est'] in [3,4], vec['rsi'] in [1], sv.hour in [6], sv.cl_1h in [2]])>=3
    #long = sv.dow in [0,1,2,3,6] and sum([vec['iv_est'] in [3,4], vec['rsi'] in [1], (sv.hour in [3,4,5,6] and vec['atr'] in [3,4]), sv.cl_1h in [2]])>=3

    

    # long = (sv.dow in [0,1,2,3,6]) and (sv.hour in [3,4,5]) and (
    # sum([
    #     vec["iv_est"] in [3,4],
    #     vec["rsi"] in [1],
    #     sv.cl_1h in [2],
    # ]) >= 2
# )



    if short and not long:
        sv.amount = 10000
        return 2
    
    if long and not short:
        sv.amount = 5000
        return 1

    return 0

 
 
def long_signal(vec: dict, sv) -> bool:
    """
    vec: dict с полями из vector.csv (iv_est, rsi, atr, squize_index, ...)
    sv: объект/namespace с полями dow, hour, cl_1h

    Возвращает True/False для лонг-сигнала.
    """

    dow_ok = sv.dow in [0, 1, 2, 3, 6]

    iv_ok = vec["iv_est"] in [3, 4]
    rsi_ok = vec["rsi"] in [1]
    cl_ok = sv.cl_1h in [2]

    # CORE — как “движок” стратегии: он у тебя самый переносимый
    core = dow_ok and iv_ok and rsi_ok and cl_ok

    # RELAX для расширенных часов — только при высокой волатильности,
    # иначе в 2025 получаются массовые таймауты и средний профит обнуляется
    atr_ok = vec["atr"] in [4]

    # опционально: ещё более консервативно (лучше защищает 2025)
    # если не хочешь — просто поставь sq_ok = True
    sq_ok = vec.get("squize_index", None) in [0, 3]

    # “ровно одно из iv_ok / rsi_ok”, при этом cl_ok обязателен
    relax = (
        dow_ok
        and (sv.hour in [4, 5, 6])
        and atr_ok
        and cl_ok
        and sq_ok
        and ((iv_ok and (not rsi_ok)) or (rsi_ok and (not iv_ok)))
    )

    return core or relax