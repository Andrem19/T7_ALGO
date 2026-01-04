from markov.hmm_regime_1h import (
    run_portability_oos_experiment_1h,
    FeatureConfig1H,
    StabilizerConfig,
    TradingRuleConfig,
)

out = run_portability_oos_experiment_1h(
    csv_path="/home/jupiter/PYTHON/MARKET_DATA/_crypto_data/BTCUSDT/BTCUSDT_1h.csv",
    out_dir="markov/test_1",
    n_states=5,
    feature_cfg=FeatureConfig1H(win_vol=48, win_sma_fast=48, win_sma_slow=240),
    stabilizer_cfg=StabilizerConfig(p_switch=0.55, gap_switch=0.05, min_dwell_h=4),
    rule=TradingRuleConfig(
        long_states=(3, 4),
        short_states=(0, 1),
        pmax_min=0.75,
        gap_min=0.06,
        trend_filter=True,
        dow_entry_filter=True,
    ),
)

print(out)  # пути к summary_oos.csv и summary.json