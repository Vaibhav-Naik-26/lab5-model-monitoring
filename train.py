from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    joblib.dump(
        {
            "model": pipeline,
            "feature_names": list(X.columns),
            "target_names": list(wine.target_names),
        },
        MODEL_DIR / "wine_model.joblib",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print(f"Saved: {MODEL_DIR / 'wine_model.joblib'}")


if __name__ == "__main__":
    main()
