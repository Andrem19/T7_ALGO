import shared_vars as sv
from loop.signal import get_signal
from loop.engine_1 import engine_1
from helpers.util import find_candle_index
import helpers.util as util


def run(h: int, start: int):
    data_len = len(sv.data_1h)-48
    i = 5000
    while i < data_len:
        
        signal = get_signal(i, start)
        
        if signal !=0:
            index_5m = find_candle_index(sv.data_1h[i][0], sv.data_5m)
            result_1 = engine_1(h, index_5m, i, signal)

            sv.positions_list.append(result_1)
            
            i+=(result_1['duration_min']//60)+1
        else:
            i+=1
        
       
    