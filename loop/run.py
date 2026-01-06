import datetime
import shared_vars as sv
from loop.quant_to_cl import quantile_classes_0_4
from loop.signal import get_signal
from loop.engine_1 import engine_1
from helpers.util import find_candle_index
import helpers.util as util
from helpers.mprint import row

PATH_TRAIN_DATA = 'ch_res_2.csv'
def run(h: int, start: int):
    data_len = len(sv.data_1h)-48
    i = 2000
    while i < data_len:
        
        signal = get_signal(i, start)
        
        if signal !=0:
            index_5m = find_candle_index(sv.data_1h[i][0], sv.data_5m)
            result_1 = engine_1(h, index_5m, i, signal)
            # result_2 = engine_1(h, index_5m, i, 2)

            sv.positions_list.append(result_1)
            sv.summa += result_1['profit']
            sv.full+=1
            color_profit = 'red' if result_1['profit'] <= 0 else 'green'
            row({'yellow': f'{sv.dt}', 'blue': f'summa: {sv.summa}', 'brown': f'all: {sv.full}', 'white': f'signal: {result_1["signal"]}', color_profit: f'profit: {result_1["profit"]}\n'})
            # print(f'dt: {sv.dt} summa: {sv.summa} all: {sv.full} signal: {result_1["signal"]} profit: {result_1["profit"]}')
            
            # save_dict = {
            #     'tm_ms': sv.timestamp,
            #     'ch_1d': sv.ch_1d,
            #     'hill': sv.hill,
            #     'regime_h': sv.reg_h,
            #     'regime_d': sv.reg_d,
            #     'dur_d': sv.dur_d,
            #     'dur_h': sv.dur_h,
            #     'd': sv.dow,
            #     'h': sv.hour,
            #     'cls_1h': int(sv.cls_1h),
            #     'cls_30m': int(sv.cls_30m),
            #     'cls_15m': int(sv.cls_15m),
            #     'cls_5m': int(sv.cls_5m),
            #     'super_cls': int(sv.super_cls),
            #     'cl_1d': int(sv.cl_1d),
            #     'cl_4h': int(sv.cl_4h),
            #     'cl_1h': int(sv.cl_1h),
            #     'cl_15m': int(sv.cl_15m),
            #     'profit_1': result_1['profit'],
            #     'profit_2': result_2['profit']
            # }
            # save_dict.update(quantile_classes_0_4(rsi=sv.rsi, atr=sv.atr, iv_est=sv.iv_est, squize_index=sv.squeeze_index)
            # util.append_dict_to_csv(row, PATH_TRAIN_DATA)
            i+=(result_1['duration_min']//60)+1
        else:
            i+=1
        
       
    