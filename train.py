import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from edm_ccm import edm_ccm_aggregate_score

def normalize_scores(score_dict):
    scores = np.array(list(score_dict.values()))
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return {k: 0.0 for k in score_dict}
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {k: 0.5 for k in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    tickers = list(score_dict.keys())
    return {tickers[i]: float(norm[i]) for i in range(len(norm))}

def run_for_window(returns, macro_df, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    if macro_df is None or macro_df.empty:
        return None
    macro_window = macro_df.loc[ret_window.index]
    if len(macro_window) < len(ret_window):
        return None
    raw_scores = {}
    for ticker in ret_window.columns:
        # Create a deterministic small variation based on ticker name
        ticker_hash = abs(hash(ticker)) % 10000
        s = edm_ccm_aggregate_score(
            ret_window[ticker].values,
            macro_window,
            ticker_hash=ticker_hash
        )
        if not np.isfinite(s):
            s = 0.0
        raw_scores[ticker] = float(s)
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "ccm_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    macro_df = dm.get_macro_data()
    if macro_df is None:
        print("Error: No macro data found.")
        return
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "macro_vars": config.MACRO_VARS,
        "embedding_dim": config.E,
        "tau": config.TAU,
        "library_fraction": config.LIB_SECTION,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, macro_df, w)
            if out:
                all_window_results.append(out)
            else:
                print(f"    Failed for window {w}")
        best_data = all_window_results[-1] if all_window_results else None
        results["universes"][uni_name] = {
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/edm_ccm_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
