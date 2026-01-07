from datetime import datetime

COIN = 'BTCUSDT'
amount = 10000
def get_path(t: str):
    return f'../MARKET_DATA/_crypto_data/{COIN}/{COIN}_{t}.csv'

regime = 0

summa = 0
full = 0
START=None
END=None

positions_list = []

data_1d = None
data_1h = None
data_4h = None
data_30m = None
data_15m = None
data_5m = None

dow = None
hour = None

feats = None
iv_est = None

cl_1d = None
cl_4h = None
cl_1h = None
cl_15m = None
cls_4h = None
cls_1h = None
cls_30m = None
cls_15m = None
cls_5m = None
super_cls = None

precalc_pic = None
cls_t = None

bundels_models_1 = None
bundels_models_2 = None
regime_model_1 = None
regime_model_2 = None
regime_model_3 = None
regime_model_4 = None


rsi = None
atr = None
risk_usd = None
squeeze_index = None
squeeze_count = None
hill = None
regime_state_1 = None
regime_state_2 = None
comb_classes = None
data_dvol = None
ctx = None

reg_h = None
reg_d = None
super_r = None
futures_1 = {}
last_positions = None

flag_1 = 0
dur_d = None
dur_h = None

ch_1d = None
timestamp = None
ch_res = None
hist_pos = None
dt = None

vix_dict = None
sp500_dict = None
fear_greed = None
fg = None
vix = None
sp500 = None