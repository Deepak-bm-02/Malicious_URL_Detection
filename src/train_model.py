
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "dataset" / "processed" / "features_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# =====================================
# 2. LOAD DATASET
# =====================================

print("Loading feature dataset...")

df = pd.read_csv(DATA_FILE)

# Select the 10 extracted features
feature_columns = [
    "url_length",
    "has_https",
    "dot_count",
    "hyphen_count",
    "special_char_count",
    "suspicious_word_count",
    "has_ip",
    "has_at_symbol",
    "subdomain_count",
    "digit_count"
]

X = df[feature_columns]
y = df["label"]

print("Total records:", len(df))
print("Number of features:", len(feature_columns))

# =====================================
# 3. TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================
# 4. DEFINE ML MODELS
# =====================================

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        ))
    ])
}

results = []
trained_models = {}

# =====================================
# 5. TRAIN AND EVALUATE
# =====================================

for name, model in models.items():

    print("\n================================")
    print("Training:", name)
    print("================================")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1 * 100
    })

    trained_models[name] = model

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        MODEL_DIR / filename
    )

    # Print classification results
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Malicious"],
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Malicious"],
        yticklabels=["Legitimate", "Malicious"]
    )

    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    image_name = name.lower().replace(" ", "_")

    plt.savefig(
        OUTPUT_DIR / f"{image_name}_confusion_matrix.png"
    )

    plt.close()

# =====================================
# 6. MODEL COMPARISON
# =====================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.round(2).to_string(index=False))

# Save comparison results
results_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

# =====================================
# 7. PERFORMANCE GRAPH
# =====================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 100)
)

plt.title("Logistic Regression vs SVM Performance")
plt.ylabel("Score (%)")
plt.xlabel("Machine Learning Models")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png"
)

plt.close()

# =====================================
# 8. FINAL RESULT
# =====================================

print("\nModel training completed successfully!")

print("\nSaved Models:")
print(MODEL_DIR)

print("\nSaved Evaluation Results:")
print(OUTPUT_DIR)

print("\nBest model by F1 Score:")

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print(best_model.to_string())
