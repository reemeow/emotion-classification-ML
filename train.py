
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from LR import LogisticRegressionScratch


train = pd.read_csv("data/train.csv")

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
plt.savefig("emotion_distribution.png")
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

threshold = 0.3
f1_scores = []
coverages = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):

    print(f"\n--- Fold {fold} ---")

    vec = CountVectorizer(min_df=2, max_features=5000)

    X_train = vec.fit_transform(X[train_idx])
    X_val = vec.transform(X[val_idx])

    y_train = y[train_idx]
    y_val = y[val_idx]

    # Calculate class weights from this fold's training data
    counts = np.bincount(y_train, minlength=4)
    class_weights = len(y_train) / (4 * counts)

    model = LogisticRegressionScratch(
        n_features=X_train.shape[1],
        n_classes=4,
        class_weights=class_weights
    )

    model.fit(
        X_train,
        y_train,
        learning_rate=1.0,
        epochs=1000
    )

    # Predict using confidence threshold
    y_pred = model.predict(X_val, threshold=threshold)

    macro_f1 = f1_score(
        y_val,
        y_pred,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0
    )

    coverage = np.mean(y_pred != -1)

    f1_scores.append(macro_f1)
    coverages.append(coverage)

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Coverage: {coverage:.2%}")

    print(classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=["Joy", "Sadness", "Anger", "Neutral"],
        zero_division=0
    ))


print("\n--- Cross-Validation Results ---")
print(f"Mean Macro F1: {np.mean(f1_scores):.4f}")
print(f"Std Macro F1: {np.std(f1_scores):.4f}")
print(f"Mean Coverage: {np.mean(coverages):.2%}")
