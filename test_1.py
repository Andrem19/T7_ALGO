# from markov.hmm import train_hmm_regime_model, FeatureConfig, load_hmm_regime_model, infer_regime_latest
from helpers.get_data_api import get_futures_klines_np
from markov.hmm import train_hmm_regime_model, load_hmm_regime_model, infer_regime_latest, current_regime_duration
import numpy as np

np.set_printoptions(precision=6, suppress=True)

n_states = 4
art = train_hmm_regime_model(
    csv_path="/home/jupiter/PYTHON/MARKET_DATA/_crypto_data/BTCUSDT/BTCUSDT_1h.csv",
    model_path=f"markov/model_{n_states}h_22/hmm_btc_1h.npz",
    out_dir=f"markov/model_{n_states}h_22/artifacts",
    start="2020-1-01",
    end="2022-1-01",          # например исключить 2025
    n_states=n_states,
    n_iter=250
)
print(art)
# model = load_hmm_regime_model("markov/model_3t/hmm_btc_1d.npz")

# data_1h = get_futures_klines_np('BTCUSDT', '1d', 300, include_incomplete_last=True)

# print(len(data_1h))

# res = infer_regime_latest(data_1h, model, lookback_days=180)
# print(res)

# dur = current_regime_duration(res, use_viterbi=True)
# print(dur)
# p = res["current_state_probs"]
# print("probs:", np.array2string(p, precision=6, suppress_small=True))
# print("probs %:", [f"{x*100:.2f}%" for x in p])

