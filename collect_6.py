import helpers.get_data as gd

import shared_vars as sv
import talib
import random
import time
import helpers.util as util
import numpy as np
np.set_printoptions(suppress=True, floatmode="fixed", precision=6)
from datetime import datetime, timezone
import vizualization.viz as viz
import helpers.predict_pic as pred
import os
import helpers.classify as clasif

sv.START=datetime(2020, 8, 1)
sv.END=datetime(2027, 1, 1)

data_1d = gd.load_data_sets(1440)
data_4h = gd.load_data_sets(240)
data_1h = gd.load_data_sets(60)
data_15m = gd.load_data_sets(15)

model_1d = pred.load_trained_model('/home/jupiter/PYTHON/T4_ALGO/_models/1d/model_epoch_012.keras')
model_4h = pred.load_trained_model('/home/jupiter/PYTHON/T4_ALGO/_models/4h/model_epoch_012.keras')
model_1h = pred.load_trained_model('/home/jupiter/PYTHON/T4_ALGO/_models/1h/model_epoch_010.keras')
model_15m = pred.load_trained_model('/home/jupiter/PYTHON/T4_ALGO/_models/15m/model_epoch_016.keras')

len_data = len(data_1h)-24
print(f'Data length: {len_data}')

for i in range(1440, len_data):
    timestamp = 1767222000000.0
    dt = datetime.fromtimestamp(data_1h[i][0]/1000)
    weekday = dt.weekday()
    if data_1h[i][0] <= timestamp:
        continue
    line_dict = {'timestamp_ms': data_1h[i][0]}
    
    #===1D===
    path_1d = '_temp_pic/1d.png'
    index = util.find_index(data_1h[i][0], data_1d)
    if index is not None and index >= 60:
        sample_1d = data_1d[index-60:index]

        last_cand = util.combine_last_candle(data_1d[index][0], data_1h[i][0], data_1h)
        if last_cand is not None:
            sample_1d = np.append(sample_1d, [last_cand], axis=0)

        viz.save_candlesticks_pic_1(sample_1d, path_1d)
    else:
        continue

    #===4H===
    path_4h = '_temp_pic/4h.png'
    index = util.find_index(data_1h[i][0], data_4h)
    if index is not None and index >= 60:
        sample_4h = data_4h[index-60:index]

        last_cand = util.combine_last_candle(data_4h[index][0], data_1h[i][0], data_1h)
        if last_cand is not None:
            sample_4h = np.append(sample_4h, [last_cand], axis=0)

        viz.save_candlesticks_pic_1(sample_4h, path_4h)
    else:
        continue

    #===1H===
    path_1h = '_temp_pic/1h.png'
    sample_1h = data_1h[i-60:i]
    viz.save_candlesticks_pic_1(sample_1h, path_1h)
    
    #===15m===
    path_15m = '_temp_pic/15m.png'
    index = util.find_index(data_1h[i][0], data_15m)
    if index is not None and index >= 60:
        sample_15m = data_15m[index-60:index]
        viz.save_candlesticks_pic_1(sample_15m, path_15m)
    else:
        continue
    
    pred_idx_1d, pred_name_1d, probs_1d = pred.predict_single_image(model_1d, path_1d, img_size=180)
    pred_idx_4h, pred_name_4h, probs_4h = pred.predict_single_image(model_4h, path_4h, img_size=180)
    pred_idx_1h, pred_name_1h, probs_1h = pred.predict_single_image(model_1h, path_1h, img_size=180)
    pred_idx_15m, pred_name_15m, probs_15m = pred.predict_single_image(model_15m, path_15m, img_size=180)
    
    line_dict['cl_1d'] = pred_idx_1d
    line_dict['cl_4h'] = pred_idx_4h
    line_dict['cl_1h'] = pred_idx_1h
    line_dict['cl_15m'] = pred_idx_15m
    
    rsi = clasif.rsi_category(sample_1h, 20)
    atr = clasif.atr_category(sample_1h, 20)
    bol = clasif.bollinger_category(sample_1h, 20)
    line_dict['rsi_20'] = rsi
    line_dict['atr_20'] = atr
    line_dict['bol_20'] = bol

    line_dict['weekday'] = weekday
    line_dict['hour'] = dt.hour
    
    body_1, upper_1, lower_1 = clasif.candle_parts_pct(data_1h[i-1])
    body_2, upper_2, lower_2 = clasif.candle_parts_pct(data_1h[i-2])
    body_3, upper_3, lower_3 = clasif.candle_parts_pct(data_1h[i-3])
    body_4, upper_4, lower_4 = clasif.candle_parts_pct(data_1h[i-4])
    
    one_cl_1 = clasif.classify_candlestick(body_1, upper_1, lower_1)
    one_cl_2 = clasif.classify_candlestick(body_2, upper_2, lower_2)
    one_cl_3 = clasif.classify_candlestick(body_3, upper_3, lower_3)
    one_cl_4 = clasif.classify_candlestick(body_4, upper_4, lower_4)
    
    two_cl_12 = clasif.classify_two_candlesticks(body_1, upper_1, lower_1, body_2, upper_2, lower_2)
    two_cl_23 = clasif.classify_two_candlesticks(body_2, upper_2, lower_2, body_3, upper_3, lower_3)
    two_cl_34 = clasif.classify_two_candlesticks(body_3, upper_3, lower_3, body_4, upper_4, lower_4)

    line_dict['one_cl_1'] = one_cl_1
    line_dict['one_cl_2'] = one_cl_2
    line_dict['one_cl_3'] = one_cl_3
    line_dict['one_cl_4'] = one_cl_4
    
    line_dict['two_cl_12'] = two_cl_12
    line_dict['two_cl_23'] = two_cl_23
    line_dict['two_cl_34'] = two_cl_34
    
    
    util.append_dict_to_csv(line_dict, 'pic_cl.csv')