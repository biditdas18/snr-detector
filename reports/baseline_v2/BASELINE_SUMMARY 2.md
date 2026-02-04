## Baseline: Direct regression to snr_score (Ridge)

Command:
python scripts/train_snr_baseline.py --input reports/anchor_features_v2.csv --alpha 100 --folds 3 --out_coef reports/baseline_v2/coef_a100_f3.csv --out_pred reports/baseline_v2/pred_a100_f3.csv

Result (n=23):
- CV R2 ≈ -0.33 (stable across folds with regularization)
- Interpretation: direct prediction of a human-composed SNR scalar is unstable in low-data regimes, motivating a decomposed scoring engine + calibration layer.
