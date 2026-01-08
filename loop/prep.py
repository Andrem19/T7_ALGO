# loop/prep.py
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from itertools import product
from typing import Dict, Any, Iterable, List
from helpers.mprint import logger
import helpers.tlg as tlg
import shared_vars as sv
import helpers.util as util
from helpers.get_data import load_data_sets
from loop.run import run
from trade_report.show import analyse_trades
from loop.search_comb import find_best_exclusion_combos
from datetime import datetime
from markov.hmm import load_hmm_regime_model
from loop.dvol_features_6am import FeatureContext
from loop.calc_comb import optimise_feature_exclusions


async def prep():
    #======================================
    #===============CONFIG=================
    #======================================
    sv.regime = 1
    sv.START=datetime(2020, 8, 1)
    sv.END=datetime(2027, 1, 1)

    
    #======================================
    #======================================
    
    # sv.data_4h = load_data_sets(240)conda activate env6
    sv.data_1d = load_data_sets(1440)
    sv.data_1h = load_data_sets(60)
    sv.data_30m = load_data_sets(30)
    sv.data_15m = load_data_sets(15)
    sv.data_5m = load_data_sets(5)
    
    # sv.data_dvol = util.load_csv_as_dict('/home/jupiter/PYTHON/MARKET_DATA/BTC_DVOL_3600s_20210101_20260106.csv', key_col="timestamp_ms", drop_constant_and_all_nan=True, drop_non_numeric=True)
    # sv.precalc_pic = util.load_csv_as_dict('pic_cl.csv', key_col="timestamp_ms", drop_constant_and_all_nan=True, drop_non_numeric=True)
    # sv.cls_t = util.load_csv_as_dict('cls_t.csv', key_col="timestamp_ms")
    # sv.regime_state_1 = util.load_csv_as_dict('markov/model_3_prod/artifacts/train_states.csv', key_col='ts_ms')
    # sv.regime_state_2 = util.load_csv_as_dict('markov/model_6_prod/artifacts/train_states.csv', key_col='ts_ms')
    # sv.vix_dict = util.load_csv_as_dict('/home/jupiter/PYTHON/MARKET_DATA/VIX_1h.csv', key_col='tm_ts')
    # sv.sp500_dict = util.load_csv_as_dict('/home/jupiter/PYTHON/MARKET_DATA/SP500_1h.csv', key_col='tm_ts')
    sv.vector = util.load_csv_as_dict('vector.csv', key_col='tm_ms')
    # sv.mvrv_nupl = util.load_csv_as_dict('/home/jupiter/PYTHON/MARKET_DATA/BTC_MVRV_NUPL_1d.csv', key_col='timestamp_ms')
    # sv.fear_greed = util.load_csv_as_dict('/home/jupiter/PYTHON/MARKET_DATA/fear_greed_1d.csv', key_col='timestamp_ms')#('/home/jupiter/PYTHON/MARKET_DATA/fear_greed_index.csv', 'timestamp_ms')
    # sv.fear_greed_stock = util.load_csv_as_dict('/home/jupiter/PYTHON/MARKET_DATA/fear_greed_index.csv', key_col='timestamp_ms')
    # sv.ctx = FeatureContext(btc_1h=sv.data_1h, dvol_1h=sv.data_dvol)
    
    
    sv.regime_model_1 = load_hmm_regime_model("markov/model_3_22/hmm_btc_1d.npz")
    sv.regime_model_2 = load_hmm_regime_model("markov/model_4h_22/hmm_btc_1h.npz")
    # sv.bundels_models_1 = load_prod_bundle("/home/jupiter/PYTHON/T5_ALGO/_models/2026-01-02/161814/catboost_forward_wf_two_stage_best_v2/prod")#("/home/jupiter/PYTHON/T5_ALGO/_models/2025-12-29/193903/catboost_forward_wf_two_stage_best_v2/prod")
    # sv.bundels_models_2 = load_prod_bundle("/home/jupiter/PYTHON/T5_ALGO/_models/2026-01-02/161937/catboost_forward_wf_two_stage_best_v2/prod")#("/home/jupiter/PYTHON/T5_ALGO/_models/2025-12-29/193939/catboost_forward_wf_two_stage_best_v2/prod")
    sv.duration = 6
    run(start=6)

    # res = find_best_exclusion_combos(
    #     sv.positions_list,
    #     max_combo_size=5,         # обычно 2
    #     min_support=30,          # чтобы не ловить случайности
    #     max_removed_trade_share=0.50,  # не предлагать правила, которые вырежут >25% сделок
    #     good_profit_penalty=0.20,      # сильнее штрафовать за вырезание хорошего
    #     top_n_print=50,
    # )
    
    
    if sv.regime == 0:
        
        out = optimise_feature_exclusions(
            sv.positions_list,
            walk_forward=True,

            wf_mode="expanding",
            wf_train_min_trades=450,
            wf_val_trades=150,
            wf_step_trades=150,
            wf_max_folds=3,

            portability_weight=2.0,
            portability_min_val_improvement=0.0,
            portability_min_val_pos_profit_ratio=0.65,

            max_rules=18,
            restarts=400,
            local_passes=10,
            swap_attempts=200,

            enable_int_combo_rules=True,
            combo_sizes=(2,),          # если хочешь меньше подгона — только пары
            progress=True,
            progress_every_restarts=10,
        )

        print(out["metrics"])
        print('------------------------------')
        print(out["rules_compact"])

    else:
        analyse_trades(sv.positions_list, save=True, save_dir="trade_report")


POWER_CFG_AGGR = dict(
    # поиск
    max_rules=80,
    restarts=80,
    rcl_size=22,
    worse_budget=10,
    local_passes=44,
    swap_attempts=500,
    seed=1,

    # ограничения
    min_keep_trades=180,
    min_keep_pos_profit_ratio=0.70,

    # ===== НОВЫЕ правила "вырезать середину" для float (агрессивнее) =====
    enable_float_ranges=True,
    float_bins=24,
    float_min_interval_trades=80,
    float_min_interval_width_frac=0.15,
    float_middle_must_include_median=True,
    float_interval_require_net_negative=True,
    float_max_intervals_per_feature=60,

    debug=False,
)