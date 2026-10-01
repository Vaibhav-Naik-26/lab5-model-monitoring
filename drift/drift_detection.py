from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_wine

OUT_DIR = Path("drift")
ALPHA = 0.05
RNG = np.random.default_rng(42)


def main() -> None:
    wine = load_wine(as_frame=True)
    reference = wine.data.copy().reset_index(drop=True)

    # Create two windows from the same underlying population so that only the
    # deliberately introduced shifts are expected to produce significant drift.
    reference_window = reference.sample(n=300, replace=True, random_state=0).reset_index(drop=True)

    current_window = reference.sample(n=300, replace=True, random_state=10).reset_index(drop=True)
    current_window["alcohol"] += 2.0
    current_window["color_intensity"] *= 1.8

    rows = []
    for feature in wine.feature_names:
        statistic, p_value = ks_2samp(reference_window[feature], current_window[feature])
        drift = "DRIFT DETECTED" if p_value < ALPHA else "No significant drift"
        rows.append(
            {
                "feature": feature,
                "test": "KS two-sample",
                "ks_statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 6),
                "alpha": ALPHA,
                "interpretation": drift,
            }
        )

    results = pd.DataFrame(rows).sort_values("p_value")
    results.to_csv(OUT_DIR / "drift_results.csv", index=False)
    reference_window.to_csv(OUT_DIR / "reference_window.csv", index=False)
    current_window.to_csv(OUT_DIR / "current_window.csv", index=False)

    print("Reference rows:", len(reference_window))
    print("Current rows:", len(current_window))
    print("\nTop drift results:")
    print(results.head(5).to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'drift_results.csv'}")


if __name__ == "__main__":
    main()
