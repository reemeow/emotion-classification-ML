
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from LR import LogisticRegressionScratch


# Directory for experiment results
RESULTS_DIR = Path("results/experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

detailed_results = []
confusion_results = []
error_results = []

train = pd.read_csv("data/train.csv")
original_count = len(train)

allowed_emotions = ["joy", "sadness", "anger", "neutral"]

train = train[
    train["emotions"].isin(allowed_emotions)
].copy()

print("\n========== DATASET FILTERING ==========")
print(f"Original samples: {original_count}")
print(f"Filtered samples: {len(train)}")
print(f"Excluded samples: {original_count - len(train)}")

print("\nClass Distribution:")
print(train["emotions"].value_counts())
print("======================================", flush=True)


# Visualize the training class distribution
emotion_counts = train["emotions"].value_counts()

plt.figure(figsize=(7, 4))
emotion_counts.plot(
    kind="bar",
    color=["skyblue", "salmon", "lightgreen", "orange"]
)
plt.title("Distribution of Emotions in Training Set")
plt.xlabel("Emotions")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "emotion_distribution.png")
plt.show()


# Encode emotion labels
label_map = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "neutral": 3
}

X = train["text"].to_numpy()
y = train["emotions"].map(label_map).to_numpy()


# Stratified 5-fold cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

thresholds = [0.3, 0.5, 0.7]
alpha = 0.65

label_names = ["Joy", "Sadness", "Anger", "Neutral"]

results = {
    t: {"f1": [], "accuracy": [], "coverage": []}
    for t in thresholds
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

    print(f"\n--- Fold {fold} ---")

    vec = TfidfVectorizer(
        min_df=2,
        max_features=5000,
        norm="l2"
    )

    X_train = vec.fit_transform(X[train_idx])
    X_val = vec.transform(X[val_idx])

    y_train = y[train_idx]
    y_val = y[val_idx]

    # Calculate class weights from this fold's training data
    counts = np.bincount(y_train, minlength=4)
    base_weights = len(y_train) / (4 * counts)

    # Adjust class weights strength
    class_weights = base_weights ** alpha

    print("\n========== CLASS WEIGHTS ==========")
    print(f"Alpha: {alpha}")

    for emotion, weight in zip(label_names, class_weights):
        print(f"{emotion}: {weight:.4f}")

    print("==================================")

    model = LogisticRegressionScratch(
        n_features=X_train.shape[1],
        n_classes=4,
        class_weights=class_weights
    )

    model.fit(
        X_train,
        y_train,
        learning_rate=1.0,
        epochs=3000
    )

    # Evaluate all thresholds on the same trained model
    for threshold in thresholds:

        y_pred = model.predict(X_val, threshold=threshold)

        if threshold == 0.3:
            probabilities = model.predict_proba(X_val)

            for i in range(len(y_val)):
                if y_pred[i] != y_val[i]:

                    predicted_label = (
                        "Uncertain"
                        if y_pred[i] == -1
                        else label_names[y_pred[i]]
                    )

                    error_results.append({
                        "Fold": fold,
                        "Text": X[val_idx][i],
                        "Actual": label_names[y_val[i]],
                        "Predicted": predicted_label,
                        "Confidence": np.max(probabilities[i]),
                        "Threshold": threshold,
                        "Alpha": alpha
                    })

        macro_f1 = f1_score(
            y_val,
            y_pred,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0
        )

        accuracy = np.mean(y_pred == y_val)
        coverage = np.mean(y_pred != -1)

        results[threshold]["f1"].append(macro_f1)
        results[threshold]["accuracy"].append(accuracy)
        results[threshold]["coverage"].append(coverage)

        report = classification_report(
            y_val,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )

        for emotion in label_names:
            detailed_results.append({
                "Fold": fold,
                "Learning Rate": 1.0,
                "Epochs": 3000,
                "Class Weights": "ON",
                "Alpha": alpha,
                "Threshold": threshold,
                "Emotion": emotion,
                "Precision": report[emotion]["precision"],
                "Recall": report[emotion]["recall"],
                "F1": report[emotion]["f1-score"],
                "Support": report[emotion]["support"],
                "Accuracy": accuracy,
                "Coverage": coverage
            })

        cm = confusion_matrix(
            y_val,
            y_pred,
            labels=[0, 1, 2, 3, -1]
        )

        all_labels = label_names + ["Uncertain"]

        for i, actual in enumerate(all_labels):
            for j, predicted in enumerate(all_labels):
                confusion_results.append({
                    "Fold": fold,
                    "Learning Rate": 1.0,
                    "Epochs": 3000,
                    "Class Weights": "ON",
                    "Alpha": alpha,
                    "Threshold": threshold,
                    "Actual": actual,
                    "Predicted": predicted,
                    "Count": cm[i, j]
                })

        print(f"\nThreshold: {threshold}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Coverage: {coverage:.2%}")


# Final results after all 5 folds
print("\n========== EXPERIMENT RESULTS ==========")

for threshold in thresholds:
    f1_values = results[threshold]["f1"]
    accuracy_values = results[threshold]["accuracy"]
    coverage_values = results[threshold]["coverage"]

    print(f"\n--- Threshold: {threshold} ---")
    print(f"Mean Macro F1: {np.mean(f1_values):.4f}")
    print(f"Std Macro F1: {np.std(f1_values):.4f}")
    print(f"Mean Accuracy: {np.mean(accuracy_values):.4f}")
    print(f"Mean Coverage: {np.mean(coverage_values):.2%}")


# Save experiment results
experiment_name = f"tfidf3000_alpha{int(round(alpha * 100)):03d}"

pd.DataFrame(detailed_results).to_csv(
    RESULTS_DIR / f"emotion_metrics_{experiment_name}.csv",
    index=False
)

pd.DataFrame(confusion_results).to_csv(
    RESULTS_DIR / f"confusion_matrices_{experiment_name}.csv",
    index=False
)

print("\nTF-IDF results saved successfully!")

pd.DataFrame(error_results).to_csv(
    RESULTS_DIR / f"error_analysis_{experiment_name}.csv",
    index=False
)

print("\nError analysis saved successfully!")
print(f"\nExperiment results saved to: {RESULTS_DIR}")
