# Emotion Classification — Logistic Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-7952B3)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-From%20Scratch-6C3483)
![Status](https://img.shields.io/badge/Status-Experimental-blueviolet)
![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?logo=huggingface&logoColor=black)

A multiclass text emotion classifier implemented with NumPy.

I started this project to understand Logistic Regression at the implementation level rather than just using a ready-made model. I built the classifier, experimented with training settings, evaluated its performance, and looked into some of its incorrect predictions.

This repository contains the implementation, training scripts, evaluation results, and a few experiments on model interpretation.

## 1. What the model does

Given a piece of English text, the model predicts one of four labels:

- Joy
- Sadness
- Anger
- Neutral

The pipeline is:

**Text → TF-IDF features → Logistic Regression → Class probabilities → Prediction**

The classifier itself is written from scratch. I use scikit-learn for feature extraction, cross-validation, and evaluation metrics.

## 2. Dataset

The training and validation data contain the following class distributions:

| Emotion | Train | Validation |
|---|---:|---:|
| Joy | 845 | 106 |
| Sadness | 815 | 84 |
| Anger | 1,021 | 107 |
| Neutral | 12,792 | 1,586 |
| **Total** | **15,473** | **1,883** |

Neutral makes up most of the dataset. This is important because a model can achieve fairly high accuracy by predicting Neutral frequently, even if it performs poorly on the other emotions.

To address this, I experimented with class weighting and used Macro F1 alongside accuracy.

The dataset files are not included in this repository. The data-loading and preparation code can be found in `load_data.py` and `prepare_data.py`.

## 3. Implementation

The Logistic Regression model is implemented in `LR.py`.

### Forward pass

The model computes linear scores:

\[
Z = XW + b
\]

Then applies softmax to convert them into probabilities:

\[
P(y=k\mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
\]

### Loss and optimization

Training uses weighted cross-entropy loss and gradient descent.

The main steps are:

1. Compute class probabilities.
2. Calculate the weighted loss.
3. Compute gradients for weights and biases.
4. Update the parameters using the learning rate.
5. Repeat for the selected number of epochs.

I also implemented a prediction threshold. If the highest predicted probability is below the threshold, the model returns `-1` (uncertain) instead of assigning an emotion.

### Feature extraction

Text is converted into TF-IDF vectors using:

| Setting | Value |
|---|---|
| `min_df` | 2 |
| `max_features` | 5,000 |
| Normalization | L2 |

## 4. Training experiments

I experimented with different learning rates, epoch counts, class-weight settings, and confidence thresholds.

I used stratified 5-fold cross-validation to compare configurations while maintaining the class distribution across folds.

The final settings were:

| Parameter | Value |
|---|---|
| Learning rate | 1.0 |
| Epochs | 3,000 |
| Class-weight exponent | 0.75 |
| Prediction threshold | 0.30 |
| Cross-validation | Stratified 5-fold |

One thing I noticed during experimentation was the trade-off between confidence threshold and coverage. A higher threshold rejected more predictions, while a lower threshold increased coverage.

## 5. Results

Final evaluation on the held-out validation set:

| Metric | Result |
|---|---:|
| Accuracy | 87.63% |
| Macro F1 | 0.6879 |
| Coverage | 99.84% |

### Per-class performance

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| Joy | 0.70 | 0.67 | 0.68 |
| Sadness | 0.58 | 0.63 | 0.61 |
| Anger | 0.55 | 0.52 | 0.54 |
| Neutral | 0.93 | 0.93 | 0.93 |

Neutral has the strongest results, which is expected given the class distribution. Anger and Sadness are more difficult for the model.

The accuracy looks relatively high, but Macro F1 gives a better picture of how the model performs across all four emotions.

The full confusion matrix, metrics, and error analysis are available in `results/final_evaluation/`.

## 6. Looking into the predictions

After evaluating the model, I wanted to understand which words were influencing its decisions.

### Feature weights

Because Logistic Regression is linear, I can inspect its learned weights to see which TF-IDF features are associated with each class.

Some of the strongest relative features were:

| Emotion | Features |
|---|---|
| Joy | glad, happy, enjoy |
| Sadness | sad, sorry, sadly |
| Anger | fuck, hate, fucking |
| Neutral | they, name, there |

These weights describe the model's learned associations, not necessarily the meaning of a word in every context.

### An incorrect prediction

One example from the error analysis:

**Text:**

> damn so the patriots are gonna get thielen after all

**Actual label:** Neutral  
**Predicted label:** Anger

I checked the contribution of individual features and then removed the word `damn` to see whether the prediction would change.

| Version | Joy | Sadness | Anger | Neutral |
|---|---:|---:|---:|---:|
| Original | 0.100 | 0.176 | 0.515 | 0.209 |
| Without `damn` | 0.138 | 0.226 | 0.262 | 0.374 |

After removing the word, the prediction changed from Anger to Neutral.

This was a useful example of how a single word can strongly influence a bag-of-words-based classifier, even when the full sentence has a different meaning.

The corresponding files are:

- `feature_importance.csv`
- `local_interpretation_anger_neutral.csv`
- `counterfactual_analysis.csv`

All are located in `results/final_evaluation/`.

## 7. Project structure

```text
emotion-classification-ML/
├── LR.py                  # Logistic Regression implementation
├── train.py               # Training and experiments
├── final_evaluation.py    # Final evaluation and analysis
├── features.py            # TF-IDF feature extraction
├── load_data.py           # Dataset loading
├── prepare_data.py        # Data preparation
├── requirements.txt
├── README.md
└── results/
    └── final_evaluation/
        ├── emotion_distribution.png
        ├── final_evaluation_metrics.csv
        ├── final_confusion_matrix.csv
        ├── final_error_analysis.csv
        ├── feature_importance.csv
        ├── local_interpretation_anger_neutral.csv
        └── counterfactual_analysis.csv
```

## 8. Running the project

Clone the repository:

```bash
git clone https://github.com/reemeow/emotion-classification-ML.git
cd emotion-classification-ML
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the required dataset files in the `data/` directory using the paths expected by the loading scripts.

Run the training experiments:

```bash
python train.py
```

Or run the final evaluation:

```bash
python final_evaluation.py
```

## 9. Notes and limitations

A few things worth keeping in mind:

- **Class imbalance:** Neutral dominates the dataset, so accuracy can be misleading.
- **Limited context:** TF-IDF does not understand word order or meaning the way a contextual language model does.
- **Ambiguous language:** Words associated with anger can appear in neutral conversations, sarcasm, or casual expressions.
- **Thresholds:** Rejecting low-confidence predictions changes coverage and may affect the interpretation of evaluation metrics.
- **Probabilities:** Softmax outputs are not automatically well-calibrated confidence estimates.

This is an experimental implementation intended for learning and analysis, not a production classifier.

## 10. Possible next steps

Some directions I may explore later:

- Compare the model against a majority-class baseline.
- Test different feature representations, such as n-grams.
- Evaluate probability calibration.
- Analyze more misclassified examples to identify recurring error patterns.
- Compare the implementation against scikit-learn's Logistic Regression as a reference.

---

Built with Python, NumPy, and scikit-learn.