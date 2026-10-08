import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.datasets import load_wine

rng = np.random.default_rng(42)
data = load_wine(as_frame=True)
X, y = data.data, data.target
saved = joblib.load("model/wine_model.joblib")
model = saved if not isinstance(saved, dict) else next(v for v in saved.values() if hasattr(v, "predict"))

# two windows of 300 rows each
ref_idx = rng.choice(len(X), 300, replace=True)
cur_idx = rng.choice(len(X), 300, replace=True)
ref_X = X.iloc[ref_idx].reset_index(drop=True)
ref_y = y.iloc[ref_idx].reset_index(drop=True)
cur_X = X.iloc[cur_idx].reset_index(drop=True)
cur_y = y.iloc[cur_idx].reset_index(drop=True)

# introduce drift in the current window
cur_X["alcohol"] = cur_X["alcohol"] + 1.0
cur_X["color_intensity"] = cur_X["color_intensity"] * 1.4

def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, 1e-4, None)
    c = np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))

rows = []
for col in X.columns:
    ks_stat, p = ks_2samp(ref_X[col], cur_X[col])
    score = psi(ref_X[col], cur_X[col])
    ks_result = "DRIFT" if p < 0.05 else "no drift"
    if score >= 0.25:
        psi_result = "major shift"
    elif score >= 0.1:
        psi_result = "moderate shift"
    else:
        psi_result = "stable"
    rows.append([col, round(ks_stat, 4), p, round(score, 4), ks_result, psi_result])

table = pd.DataFrame(rows, columns=["feature", "ks_stat", "ks_p_value", "psi", "ks_result", "psi_result"])
table = table.sort_values("ks_stat", ascending=False)
print("\nFEATURE DRIFT (KS test and PSI)")
print(table.to_string(index=False))

# prediction drift: compare predicted class counts
ref_pred = model.predict(ref_X)
cur_pred = model.predict(cur_X)
counts = np.array([
    [int(np.sum(ref_pred == k)) for k in [0, 1, 2]],
    [int(np.sum(cur_pred == k)) for k in [0, 1, 2]],
])
chi2, p_chi, _, _ = chi2_contingency(counts)
print("\nPREDICTION DRIFT (chi-square on predicted class counts)")
print("reference counts (class 0,1,2):", counts[0].tolist())
print("current counts   (class 0,1,2):", counts[1].tolist())
print("chi2 =", round(chi2, 3), " p-value =", p_chi)
print("Result:", "DRIFT" if p_chi < 0.05 else "no drift")

# quality check using true labels
ref_acc = float((ref_pred == ref_y).mean())
cur_acc = float((cur_pred == cur_y).mean())
print("\nACCURACY")
print("reference window:", round(ref_acc, 4))
print("current window:  ", round(cur_acc, 4))

table.to_csv("drift/drift_full_results.csv", index=False)
print("\nSaved: drift/drift_full_results.csv")
