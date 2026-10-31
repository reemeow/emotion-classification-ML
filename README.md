
# Emotion Classification — Logistic Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-7952B3)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-From%20Scratch-6C3483)
![Status](https://img.shields.io/badge/Status-Experimental-blueviolet)
![Hugging Face Datasets](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?logo=huggingface&logoColor=black)

A multiclass text emotion classifier implemented with NumPy.

I started this project to understand Logistic Regression at the implementation level rather than just using a ready-made model. I built the classifier, experimented with training settings, evaluated its performance, and investigated some of its incorrect predictions.

This repository contains the model implementation, training scripts, evaluation results, and experiments on model interpretation.

## 1. What the Model Does

Given a piece of English text, the model predicts one of four labels:

- Joy
- Sadness
- Anger
- Neutral

The pipeline is:

**Text → TF-IDF Features → Logistic Regression → Class Probabilities → Prediction**

The classifier itself is implemented from scratch. Scikit-learn is used for feature extraction, cross-validation, and evaluation metrics.

## 2. Dataset

This project uses the [GoEmotions dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions) from Google Research, loaded through Hugging Face using the `simplified` configuration.

GoEmotions contains Reddit comments annotated with emotion labels. I selected four classes for this project: **Joy, Sadness, Anger, and Neutral**.

### Data Preparation

The dataset is prepared in `prepare_data.py` using the following steps:

1. Keep only examples with exactly one emotion label.
2. Filter the data to the four selected emotions.
3. Convert text to lowercase, normalize whitespace, and remove empty text.
4. Remove texts associated with conflicting emotion labels within each split.
5. Remove duplicate texts within each split.
6. Remove overlapping texts between training, validation, and test sets to reduce data leakage.

The cleaned data is saved as three CSV files: `train.csv`, `val.csv`, and `test.csv`.

### Class Distribution

| Emotion | Training | Validation |
|---|---:|---:|
| Joy | 845 | 106 |
| Sadness | 815 | 84 |
| Anger | 1,021 | 107 |
| Neutral | 12,792 | 1,586 |
| **Total** | **15,473** | **1,883** |

Neutral makes up most of the dataset, creating a significant class imbalance. A model can achieve relatively high accuracy by predicting Neutral frequently, even if it performs poorly on the other emotions.

To address this, I experimented with class weighting and used Macro F1 alongside accuracy.

The dataset files are excluded from this repository, but they can be regenerated using:

```bash
python prepare_data.py
```

Dataset inspection is handled in `load_data.py`, while data cleaning and preparation are handled in `prepare_data.py`.

## 3. Implementation

The Logistic Regression model is implemented in `LR.py`.

### Forward Pass

The model computes linear scores:

$$
Z = XW + b
$$

These scores are converted into class probabilities using softmax:

$$
P(y=k\mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

### Loss and Optimization

Training uses weighted cross-entropy loss and gradient descent.

The training process consists of:

1. Computing class probabilities.
2. Calculating the weighted loss.
3. Computing gradients for weights and biases.
4. Updating the parameters using the learning rate.
5. Repeating the process for the selected number of epochs.

I also implemented a confidence threshold. If the highest predicted probability is below the threshold, the model returns `-1` (uncertain) instead of assigning an emotion.

### Feature Extraction

Text is converted into TF-IDF vectors using the following settings:

| Parameter | Value |
|---|---|
| `min_df` | 2 |
| `max_features` | 5,000 |
| Normalization | L2 |

## 4. Training Experiments

I experimented with different learning rates, epoch counts, class-weight settings, and confidence thresholds.

I used stratified 5-fold cross-validation to compare configurations while maintaining the class distribution across folds.

### Final Configuration

| Parameter | Value |
|---|---|
| Learning rate | 1.0 |
| Epochs | 3,000 |
| Class-weight exponent | 0.75 |
| Prediction threshold | 0.30 |
| Cross-validation | Stratified 5-fold |

One thing I noticed during experimentation was the trade-off between confidence threshold and coverage. A higher threshold rejected more predictions, while a lower threshold increased coverage.

### Experiment Log

I kept an Excel workbook to track the experiments, including hyperparameter settings, cross-validation results, and per-class evaluation metrics.

The workbook includes an experiment index and separate sheets for different runs. This makes it easier to compare configurations and revisit earlier results.

Some older experiments have incomplete parameter records.

**[View Experiment Log](results/experiments/Emotion_Classification_Experiments.xlsx)**

## 5. Results

The final model was evaluated on a held-out validation set.

| Metric | Result |
|---|---:|
| Accuracy | 87.63% |
| Macro F1 | 0.6879 |
| Coverage | 99.84% |

### Per-Class Performance

| Emotion | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Joy | 0.70 | 0.67 | 0.68 |
| Sadness | 0.58 | 0.63 | 0.61 |
| Anger | 0.55 | 0.52 | 0.54 |
| Neutral | 0.93 | 0.93 | 0.93 |

Neutral has the strongest results, which is consistent with the class distribution. Anger and Sadness remain more difficult for the model.

Although overall accuracy is relatively high, Macro F1 provides a better picture of performance across all four classes.

The full confusion matrix, evaluation metrics, and error analysis are available in `results/final_evaluation/`.

## 6. Looking Into the Predictions

After evaluating the model, I wanted to understand which words were influencing its decisions.

### Feature Weights

Because Logistic Regression is a linear model, I can inspect its learned weights to identify features associated with each emotion.

Some of the strongest relative features were:

| Emotion | Features |
|---|---|
| Joy | glad, happy, enjoy |
| Sadness | sad, sorry, sadly |
| Anger | fuck, hate, fucking |
| Neutral | they, name, there |

These weights describe the model's learned associations, not necessarily the meaning of a word in every context.

### An Incorrect Prediction

One example from the error analysis:

**Text:**

> damn so the patriots are gonna get thielen after all

**Actual label:** Neutral

**Predicted label:** Anger

I examined the contribution of individual features and then removed the word `damn` to see whether the prediction would change.

| Version | Joy | Sadness | Anger | Neutral |
|---|---:|---:|---:|---:|
| Original | 0.100 | 0.176 | 0.515 | 0.209 |
| Without `damn` | 0.138 | 0.226 | 0.262 | 0.374 |

After removing the word, the prediction changed from Anger to Neutral.

This example shows how a single word can strongly influence a TF-IDF-based linear classifier, even when the overall context suggests a different interpretation.

The corresponding analysis files are:

- `feature_importance.csv`
- `local_interpretation_anger_neutral.csv`
- `counterfactual_analysis.csv`

All three files are located in `results/final_evaluation/`.

This is one illustrative example rather than a general conclusion about all similar sentences.

## 7. Project Structure

```text
emotion-classification-ML/
├── LR.py                  # Logistic Regression implementation
├── train.py               # Training and experiments
├── final_evaluation.py    # Final evaluation and analysis
├── features.py            # TF-IDF feature extraction
├── load_data.py           # Dataset inspection
├── prepare_data.py        # Data preparation
├── requirements.txt
├── README.md
└── results/
    ├── experiments/
    │   └── Emotion_Classification_Experiments.xlsx
    └── final_evaluation/
        ├── emotion_distribution.png
        ├── final_evaluation_metrics.csv
        ├── final_confusion_matrix.csv
        ├── final_error_analysis.csv
        ├── feature_importance.csv
        ├── local_interpretation_anger_neutral.csv
        └── counterfactual_analysis.csv
```

## 8. Running the Project

### Clone the Repository

```bash
git clone https://github.com/reemeow/emotion-classification-ML.git
cd emotion-classification-ML
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Prepare the Dataset

Run the data preparation script:

```bash
python prepare_data.py
```

This downloads the GoEmotions dataset, applies the filtering and cleaning steps, and creates:

```text
data/
├── train.csv
├── val.csv
└── test.csv
```

The dataset files are excluded from the repository.

### Run Training Experiments

```bash
python train.py
```

### Run Final Evaluation

```bash
python final_evaluation.py
```

The evaluation results are saved in `results/final_evaluation/`.

## 9. Notes and Limitations

A few things worth keeping in mind:

- **Class imbalance:** Neutral dominates the dataset, so accuracy alone can be misleading.
- **Limited context:** TF-IDF does not capture contextual meaning in the same way as a language model.
- **Ambiguous language:** Words associated with anger can appear in neutral conversations, sarcasm, or casual expressions.
- **Confidence thresholds:** Rejecting low-confidence predictions affects coverage and how evaluation metrics should be interpreted.
- **Probability calibration:** Softmax probabilities are not automatically calibrated confidence estimates.

This is an experimental implementation intended for learning and analysis, not a production-ready emotion classifier.

## 10. Possible Next Steps

Some directions I may explore later:

- Compare the model against a majority-class baseline.
- Test different feature representations, such as n-grams.
- Evaluate probability calibration.
- Analyze more misclassified examples to identify recurring error patterns.
- Compare the implementation against scikit-learn's Logistic Regression as a reference.

---

Built with Python, NumPy, and scikit-learn.
