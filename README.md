# Empirical Dynamic Modelling (EDM) – Convergent Cross Mapping (CCM) for ETFs

Applies Convergent Cross Mapping (Sugihara 2012) to detect nonlinear causal influence from macro variables (VIX, DXY, yields) to ETF returns. Uses simplex projection on time‑delayed embeddings. The per‑ETF score is the average cross‑map skill across all macro variables – a measure of how well macro dynamics drive the ETF.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- All available macro variables (VIX, DXY, full yield curve)
- Time‑delayed embedding (E=3, τ=1)
- Library/prediction split (75/25)
- Score = correlation between observed and predicted returns (skill)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-edm-ccm-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slow due to simplex search; reduce `LIB_SECTION` for speed)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High CCM skill → macro variables causally drive the ETF (nonlinear relationship).
- Low skill → ETF is driven by other factors.

## Requirements

See `requirements.txt`.
