import shared_vars as sv
import helpers.cls.regime_predict as rpred
from helpers.get_data import load_data_sets
from helpers.util import find_candle_index
import helpers.util as util
from datetime import datetime

PATH_TRAIN_DATA = 'cls_t.csv'

def main():
    sv.bundels_models_1 = rpred.load_bundle()
    sv.START=datetime(2020, 8, 1)
    sv.END=datetime(2027, 1, 1)
    sv.data_1h = load_data_sets(60)
    sv.data_30m = load_data_sets(30)
    sv.data_15m = load_data_sets(15)
    sv.data_5m = load_data_sets(5)
    
    for i in range(60, len(sv.data_1h)-1):
        timestamp = 1765490400000.0
        dt = datetime.fromtimestamp(sv.data_1h[i][0]/1000)
        if sv.data_1h[i][0] <= timestamp:
            continue
        data_1h = sv.data_1h[i-60:i]
    
        index_30m = find_candle_index(sv.data_1h[i][0], sv.data_30m)
        data_30m = sv.data_30m[index_30m-60:index_30m]

        index_15m = find_candle_index(sv.data_1h[i][0], sv.data_15m)
        data_15m = sv.data_15m[index_15m-60:index_15m]
        
        index_5m = find_candle_index(sv.data_1h[i][0], sv.data_5m)
        data_5m = sv.data_5m[index_5m-60:index_5m]
        sv.super_cls, sv.regimes, super_desc = rpred.predict_superstate({"1h": data_1h, "30m": data_30m, "15m": data_15m, "5m": data_5m}, sv.bundels_models_1, return_description=True)

        row = {}
        
        row['timestamp_ms'] = sv.data_1h[i][0]
        row['super_cls'] = sv.super_cls
        row.update(sv.regimes)
        
        util.append_dict_to_csv(row, PATH_TRAIN_DATA)
        
        
if __name__ == "__main__":
    main()
