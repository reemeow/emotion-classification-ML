
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from LR import LogisticRegressionScratch


# Directory for final evaluation results
FINAL_DIR = Path("results/final_evaluation")
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# Load training and validation datasets
train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")

allowed_emotions = ["joy", "sadness", "anger", "neutral"]

train = train[
    train["emotions"].isin(allowed_emotions)
].copy()

val = val[
    val["emotions"].isin(allowed_emotions)
].copy()


# Encode emotion labels
label_map = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "neutral": 3
}

label_names = ["Joy", "Sadness", "Anger", "Neutral"]

X_train_text = train["text"].to_numpy()
y_train = train["emotions"].map(label_map).to_numpy()

X_val_text = val["text"].to_numpy()
y_val = val["emotions"].map(label_map).to_numpy()


# Print dataset information
print("\n========== DATASET INFORMATION ==========")
print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")

print("\nTraining Class Distribution:")
print(train["emotions"].value_counts())

print("\nValidation Class Distribution:")
print(val["emotions"].value_counts())


# TF-IDF feature extraction
vec = TfidfVectorizer(
    min_df=2,
    max_features=5000,
    norm="l2"
)

X_train = vec.fit_transform(X_train_text)
X_val = vec.transform(X_val_text)


# Calculate class weights from training data
counts = np.bincount(y_train, minlength=4)
base_weights = len(y_train) / (4 * counts)

alpha = 0.75
class_weights = base_weights ** alpha

print("\n========== CLASS WEIGHTS ==========")
for emotion, weight in zip(label_names, class_weights):
    print(f"{emotion}: {weight:.4f}")


# Train final model
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


# Final evaluation on validation data
threshold = 0.3

y_pred = model.predict(
    X_val,
    threshold=threshold
)

probabilities = model.predict_proba(X_val)

accuracy = accuracy_score(y_val, y_pred)

macro_f1 = f1_score(
    y_val,
    y_pred,
    labels=[0, 1, 2, 3],
    average="macro",
    zero_division=0
)

coverage = np.mean(y_pred != -1)

print("\n========== FINAL EVALUATION ==========")
print(f"Alpha: {alpha}")
print(f"Threshold: {threshold}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Coverage: {coverage:.2%}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=label_names,
        zero_division=0
    )
)


# Confusion matrix including Uncertain predictions
all_labels = label_names + ["Uncertain"]

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[0, 1, 2, 3, -1]
)

cm_df = pd.DataFrame(
    cm,
    index=all_labels,
    columns=all_labels
)

print("\nConfusion Matrix:")
print(cm_df)

cm_df.to_csv(
    FINAL_DIR / "final_confusion_matrix.csv"
)


# Save final evaluation metrics
report = classification_report(
    y_val,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=label_names,
    output_dict=True,
    zero_division=0
)

metrics_results = []

for emotion in label_names:
    metrics_results.append({
        "Emotion": emotion,
        "Precision": report[emotion]["precision"],
        "Recall": report[emotion]["recall"],
        "F1": report[emotion]["f1-score"],
        "Support": report[emotion]["support"],
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Coverage": coverage,
        "Alpha": alpha,
        "Threshold": threshold
    })

pd.DataFrame(metrics_results).to_csv(
    FINAL_DIR / "final_evaluation_metrics.csv",
    index=False
)


# Save validation prediction errors
error_results = []

for i in range(len(y_val)):
    if y_pred[i] != y_val[i]:

        predicted_label = (
            "Uncertain"
            if y_pred[i] == -1
            else label_names[y_pred[i]]
        )

        error_results.append({
            "Text": X_val_text[i],
            "Actual": label_names[y_val[i]],
            "Predicted": predicted_label,
            "Confidence": np.max(probabilities[i]),
            "Threshold": threshold
        })

pd.DataFrame(
    error_results,
    columns=[
        "Text",
        "Actual",
        "Predicted",
        "Confidence",
        "Threshold"
    ]
).to_csv(
    FINAL_DIR / "final_error_analysis.csv",
    index=False
)

print("\nFinal evaluation files saved successfully!")
print(f"Results directory: {FINAL_DIR}")

# Model Interpretation: Top positive and negative features

feature_names = vec.get_feature_names_out()
weights = model.W

interpretation_results = []

for class_idx, emotion in enumerate(label_names):

    # Compare each feature's weight with its average weight
    # across all classes
    relative_weights = (
        weights[:, class_idx] - weights.mean(axis=1)
    )

    top_positive = np.argsort(relative_weights)[-20:][::-1]
    top_negative = np.argsort(relative_weights)[:20]

    print(f"\n========== {emotion.upper()} ==========")

    print("\nTop 20 Positive Features:")
    for idx in top_positive:
        print(f"{feature_names[idx]}: {relative_weights[idx]:.4f}")

        interpretation_results.append({
            "Emotion": emotion,
            "Feature": feature_names[idx],
            "Weight": relative_weights[idx],
            "Type": "Positive"
        })

    print("\nTop 20 Negative Features:")
    for idx in top_negative:
        print(f"{feature_names[idx]}: {relative_weights[idx]:.4f}")

        interpretation_results.append({
            "Emotion": emotion,
            "Feature": feature_names[idx],
            "Weight": relative_weights[idx],
            "Type": "Negative"
        })


# Save interpretation results
pd.DataFrame(interpretation_results).to_csv(
    FINAL_DIR / "feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")


# Local Interpretation: Anger vs Neutral

anger_idx = label_map["anger"]
neutral_idx = label_map["neutral"]

# Find a validation error where Neutral was predicted as Anger
selected_idx = next(
    (
        i for i in range(len(y_val))
        if y_val[i] == neutral_idx
        and y_pred[i] == anger_idx
    ),
    None
)

if selected_idx is not None:

    text = X_val_text[selected_idx]
    x_row = X_val[selected_idx]

    feature_names = vec.get_feature_names_out()

    # Contribution to Anger score relative to Neutral
    weight_difference = (
        model.W[:, anger_idx] -
        model.W[:, neutral_idx]
    )

    contributions = (
        x_row.toarray().ravel() * weight_difference
    )

    active_features = x_row.nonzero()[1]

    explanation = pd.DataFrame({
        "Feature": feature_names[active_features],
        "TF-IDF": x_row.toarray().ravel()[active_features],
        "Contribution": contributions[active_features]
    })

    explanation = explanation.sort_values(
        "Contribution",
        ascending=False
    )

    print("\n========== LOCAL INTERPRETATION ==========")
    print(f"Text: {text}")
    print("Actual: Neutral")
    print("Predicted: Anger")

    print("\nWord Contributions:")
    print(explanation.to_string(index=False))

    bias_difference = (
        model.b[anger_idx] -
        model.b[neutral_idx]
    )

    print(f"\nBias Difference: {bias_difference:.4f}")
    print(
        "Anger - Neutral Score Difference: "
        f"{bias_difference + contributions.sum():.4f}"
    )

    explanation.to_csv(
        FINAL_DIR / "local_interpretation_anger_neutral.csv",
        index=False
    )

else:
    print("No Neutral-to-Anger error found.")



# Counterfactual Analysis: Removing "damn"

original_text = (
    "damn so the patriots are gonna get thielen after all"
)

modified_text = (
    "so the patriots are gonna get thielen after all"
)

test_texts = [original_text, modified_text]

# Transform using the existing fitted TF-IDF vectorizer
X_counterfactual = vec.transform(test_texts)

# Get predictions and probabilities
cf_predictions = model.predict(
    X_counterfactual,
    threshold=0.3
)

cf_probabilities = model.predict_proba(
    X_counterfactual
)

emotion_names = ["Joy", "Sadness", "Anger", "Neutral"]

counterfactual_results = []

for i, text in enumerate(test_texts):

    pred_idx = int(cf_predictions[i])

    predicted_emotion = (
        emotion_names[pred_idx]
        if pred_idx != -1
        else "Uncertain"
    )

    result = {
        "Version": "Original" if i == 0 else "Modified",
        "Text": text,
        "Prediction": predicted_emotion,
        "Confidence": float(np.max(cf_probabilities[i]))
    }

    for class_idx, emotion in enumerate(emotion_names):
        result[f"{emotion}_Probability"] = float(
            cf_probabilities[i, class_idx]
        )

    counterfactual_results.append(result)

    print(f"\n========== {result['Version']} ==========")
    print(f"Text: {text}")
    print(f"Prediction: {predicted_emotion}")
    print(f"Confidence: {result['Confidence']:.4f}")

    for class_idx, emotion in enumerate(emotion_names):
        print(
            f"{emotion}: "
            f"{cf_probabilities[i, class_idx]:.4f}"
        )


# Save counterfactual results
counterfactual_df = pd.DataFrame(counterfactual_results)

counterfactual_df.to_csv(
    FINAL_DIR / "counterfactual_analysis.csv",
    index=False
)

print("\nCounterfactual analysis saved successfully!")
