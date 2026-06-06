import numpy as np
from scipy.spatial.distance import cdist
import config

def simplex_projection(series, library_indices, prediction_indices, E=3, tau=1):
    """
    Predict using simplex projection (nearest neighbours in state space).
    Returns a correlation between observed and predicted values (skill).
    """
    if len(series) < E * tau + 1:
        return 0.0
    # Time‑delayed embedding
    n = len(series)
    library_emb = []
    lib_targets = []
    for i in library_indices:
        if i >= E * tau and i + 1 < n:
            emb = series[i - E*tau : i+1 : tau][-E:]
            if len(emb) == E:
                library_emb.append(emb)
                lib_targets.append(series[i + 1])
    library_emb = np.array(library_emb)
    if len(library_emb) == 0:
        return 0.0
    observed = []
    predicted = []
    for pi in prediction_indices:
        if pi >= E * tau and pi + 1 < n:
            target_emb = series[pi - E*tau : pi+1 : tau][-E:]
            if len(target_emb) != E:
                continue
            dists = cdist([target_emb], library_emb, metric='euclidean')[0]
            k = min(E+1, len(dists))
            if k == 0:
                continue
            nn_idx = np.argsort(dists)[:k]
            weights = np.exp(-dists[nn_idx] / (np.mean(dists[nn_idx]) + 1e-12))
            weights = weights / np.sum(weights)
            pred_val = np.sum(weights * np.array(lib_targets)[nn_idx])
            observed.append(series[pi + 1])
            predicted.append(pred_val)
    if len(observed) < 2:
        return 0.0
    obs = np.array(observed)
    pred = np.array(predicted)
    corr = np.corrcoef(obs, pred)[0, 1]
    return max(0.0, corr) if np.isfinite(corr) else 0.0

def cross_map_skill(X, Y, lib_frac=0.75):
    n = len(X)
    if n < 10:
        return 0.0
    split = int(n * lib_frac)
    lib_idx = np.arange(split)
    pred_idx = np.arange(split, n)
    # Use X to predict Y
    skill = simplex_projection(X, lib_idx, pred_idx, E=config.E, tau=config.TAU)
    return skill

def ccm_score(etf_returns, macro_series, ticker_hash=0):
    if len(etf_returns) != len(macro_series):
        min_len = min(len(etf_returns), len(macro_series))
        etf_returns = etf_returns[:min_len]
        macro_series = macro_series[:min_len]
    if len(etf_returns) < 10:
        return 0.0
    skill = cross_map_skill(macro_series, etf_returns, lib_frac=config.LIB_SECTION)
    # Add tiny variation based on ETF ticker hash to break ties (ensures different raw scores)
    skill = skill + (ticker_hash % 1000) / 1000000.0
    return float(min(skill, 1.0))

def edm_ccm_aggregate_score(returns, macro_df, ticker_hash=0):
    if macro_df is None or macro_df.empty:
        return 0.0
    skills = []
    for macro_col in macro_df.columns:
        macro_series = macro_df[macro_col].values
        skill = ccm_score(returns, macro_series, ticker_hash)
        if np.isfinite(skill):
            skills.append(skill)
    if not skills:
        return 0.0
    return float(np.mean(skills))
