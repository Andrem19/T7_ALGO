import shared_vars as sv
from datetime import datetime
import time
import math
import helpers.util as util
import math
import copy
from loop.quant_to_cl import quantile_classes_0_4
from helpers.mprint import red

FEE_RATE = 0.0006
PATH_TRAIN_DATA = '/home/jupiter/PYTHON/T5_ALGO/_train_data/train_set_71_full.csv'


def engine_1(h, i, gen_ind, signal):
    timefinish = h*12
    open_price = sv.data_5m[i][1]
    fut_qty = sv.amount/open_price
    
    trashhold_tp =  0.05

    trashhold_sl = 0.035 if signal == 1 else 0.025

    sl = open_price * (1+trashhold_sl) if signal == 2 else open_price * (1-trashhold_sl)
    tp = open_price * (1-trashhold_tp) if signal == 2 else open_price * (1+trashhold_tp)
    
    ind = i
    dt = datetime.fromtimestamp(sv.data_5m[i][0]/1000)

    type_of_close = ''
    close_price = 0

    higest_point = -math.inf
    
    
    
    while True:
        duration = ind - i
        
        if (sv.data_5m[ind][2] > sl and signal == 2) or (sv.data_5m[ind][3] < sl and signal == 1):
            type_of_close = f'sl_{signal}'
            close_price = sl
            break
        if (sv.data_5m[ind][3] < tp  and signal == 2) or (sv.data_5m[ind][2] > tp and signal == 1):
            type_of_close = f'tp_{signal}'
            close_price = tp
            break

        if duration >= timefinish:
            type_of_close = 'timefinish'
            close_price = sv.data_5m[ind][4]
            break
        
        #======CONDITION CLOSE===========
        
        if sv.data_5m[ind][4] > higest_point:
            higest_point=sv.data_5m[ind][4]
            if duration*5 > 550 and signal == 2:
                type_of_close = 'max_point_1'
                close_price = sv.data_5m[ind][4]
                break
            

        last_20m = util.calculate_percent_difference(sv.data_5m[ind-3][1], sv.data_5m[ind][4])
        if (last_20m > 0.02) and signal == 1:
            type_of_close = 'vol_p'
            close_price = sv.data_5m[ind][4]
            break
        
        
        ind += 1
    
    fut_profit = (open_price-close_price) * fut_qty if signal == 2 else (close_price-open_price) * fut_qty
    fee_fut = (open_price * fut_qty * FEE_RATE)
    fut_profit -= fee_fut
    
    result = {
        'fee': 0,
        'open_price': open_price,
        'close_price': close_price,
        'open_time': dt.timestamp(),
        'signal': signal,
        'type_of_close': type_of_close,
        'duration_min': duration*5,
        'profit': fut_profit,
        'fee': fee_fut,
        'features': {
            'feer_and_greed': sv.fg,
            'sp500': sv.sp500,
            'vix': sv.vix,
            'hill': sv.hill,
            'rsi': float(sv.rsi),
            'atr': float(sv.atr),
            'iv_est': float(sv.iv_est),
            'squize_index': float(sv.squeeze_index),
            # 'ret_6h': float(sv.feats['ret_6h']),
            # 'dvol_minus_rv_12h': float(sv.feats['dvol_minus_rv_12h']),
            # 'atr_ratio_24h_7d': float(sv.feats['atr_ratio_24h_7d']),
            'regime_h': sv.reg_h,
            'regime_d': sv.ch_1d,
            # 'dur_d': sv.dur_d,
            # 'dur_h': sv.dur_h,
            'd': dt.weekday(),
            # 'super_r': sv.super_r,
            # 'cls_1h': int(sv.cls_1h),
            # 'cls_30m': int(sv.cls_30m),
            # 'cls_15m': int(sv.cls_15m),
            # 'cls_5m': int(sv.cls_5m),
            # 'super_cls': int(sv.super_cls),
            # 'd': sv.dow,
            'h': sv.hour,
            # 'cl_1d': int(sv.cl_1d),
            'cl_4h': int(sv.cl_4h),
            # 'cl_1h': int(sv.cl_1h),
            # 'cl_15m': int(sv.cl_15m),
            
        }
    }
    # result['features'].update(sv.comb_classes)
    # result['features'].update(quantile_classes_0_4(rsi=sv.rsi, atr=sv.atr, iv_est=sv.iv_est, squize_index=sv.squeeze_index)
    

    
    # print(f'{dt} signal: {signal} open: {open_price} close: {close_price}')
    # print('Duration min: ', duration*5)
    # print('Profit: ', result['profit'])
    # print(f'Summa: {sv.summa} full: {sv.full}\n\n')
    
    # row = copy.deepcopy(result['features'])
    # row['targ_1'] = 1 if  fut_profit > 0 else 0
    # row['targ_2'] = fut_profit / sv.risk_usd
    # util.append_dict_to_csv(row, PATH_TRAIN_DATA)

    
    return result