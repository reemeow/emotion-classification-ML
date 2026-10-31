# Emotion Classification with Logistic Regression from Scratch

**A multiclass text emotion classification project focused on understanding machine learning algorithms, optimization, and model evaluation.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-013243?logo=numpy&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-7952B3)
![Status](https://img.shields.io/badge/Status-Experimental-blueviolet)

---

## Overview
This project explores the mathematical foundations of multiclass Logistic Regression by implementing the classification algorithm from scratch using Python and NumPy.
The model classifies text into four emotion categories: **Joy, Sadness, Anger, and Neutral**.
Rather than relying on a pre-built Logistic Regression classifier, the project focuses on implementing Softmax, Weighted Cross-Entropy Loss, and Gradient Descent, while investigating the challenges of imbalanced datasets.

## Project Structure
The project is organized into separate Python modules for data preparation, feature extraction, model implementation, and training.

emotion-classification-ML/
│
├── LR.py
├── load_data.py
├── prepare_data.py
├── features.py
├── train.py
└── emotion_distribution.png

## Main Components

LR.py: Implements multiclass Logistic Regression, including Softmax, Weighted Cross-Entropy Loss, Gradient Descent, and prediction.

load_data.py: Handles loading the dataset.

prepare_data.py: Prepares the data for model training and evaluation.

features.py: Converts text into numerical features using CountVectorizer.

train.py: Runs model training and evaluation experiments.

emotion_distribution.png: Visualizes the distribution of emotion classes in the dataset.

## Emotion Distribution
The dataset contains four emotion classes with an imbalanced distribution, where Neutral is the majority class.

