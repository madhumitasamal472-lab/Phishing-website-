"""
Phishing Website Detection using Machine Learning
Author: Madhusmita Samal

Beginner-friendly project using the UCI Phishing Websites dataset.
Important: This model predicts from the dataset's 30 pre-extracted website
features. It does NOT accept a raw URL string and independently inspect a site.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")


def load_dataset():
    """Download the public UCI dataset and return feature and target tables."""
    print("Downloading the UCI Phishing Websites dataset...")
    dataset = fetch_ucirepo(id=327)

    X = dataset.data.features.copy()
    target_table = dataset.data.targets.copy()
    if target_table.shape[1] == 0:
        raise ValueError("The dataset did not provide a target column.")

    y = target_table.iloc[:, 0].copy()
    y.name = "class"

    # UCI dataset labels: -1 = phishing, 0 = suspicious, 1 = legitimate.
    # Convert numeric labels to readable names when possible.
    numeric_y = pd.to_numeric(y, errors="coerce")
    if numeric_y.notna().all():
        label_map = {-1: "Phishing", 0: "Suspicious", 1: "Legitimate"}
        y = numeric_y.map(label_map).fillna(numeric_y.astype(str))
    else:
        y = y.astype(str).str.strip()

    # Keep rows with valid targets and replace missing feature values safely.
    valid = y.notna()
    X = X.loc[valid].copy()
    y = y.loc[valid].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if X.empty:
        raise ValueError("The dataset is empty after basic cleaning.")

    print(f"Rows available: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts())
    return X, y


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (class order shown below):")
    print(list(model.classes_))
    print(confusion_matrix(y_test, predictions, labels=model.classes_))

    # Save model and the exact feature order required for later predictions.
    model_path = MODEL_DIR / "phishing_detection_model.joblib"
    joblib.dump(
        {"model": model, "feature_names": list(X.columns)},
        model_path,
    )
    print(f"\nSaved trained model to: {model_path}")

    # Save a machine-readable evaluation summary.
    summary = pd.DataFrame(
        {
            "metric": ["test_accuracy", "training_rows", "testing_rows", "feature_count"],
            "value": [accuracy, len(X_train), len(X_test), X.shape[1]],
        }
    )
    summary.to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)

    # Plot the ten most important features.
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    importance.sort_values().plot(kind="barh")
    plt.title("Top 10 Feature Importances — Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved feature importance chart to: {plot_path}")

    # Demonstrate prediction on one held-out row (not a manually typed URL).
    example = X_test.iloc[[0]]
    actual = y_test.iloc[0]
    predicted = model.predict(example)[0]
    print("\n========== SAMPLE TEST RECORD ==========")
    print(f"Actual dataset class: {actual}")
    print(f"Predicted class: {predicted}")
    print(
        "Note: This is a record of pre-extracted dataset features, "
        "not a live URL scan."
    )

    print("\nProject completed. Review the metrics before reporting results.")


if __name__ == "__main__":
    main()
