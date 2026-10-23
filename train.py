import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from LR import LogisticRegressionScrtach
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")


print("Train shape:", train.shape)
vec = CountVectorizer(min_df=2, max_features=5000)
X_train = vec.fit_transform(train['text'])

label_map = {"joy": 0, "sadness": 1, "anger": 2, "neutral": 3}
y_train = train['emotions'].map(label_map).to_numpy()
counts =np.bincount(y_train)
print("class coounts:", counts)
clas_weights=len(y_train)/ (4*counts)
print("class weights:", clas_weights)                           
model = LogisticRegressionScrtach(n_features=5000, n_classes=4)
model.fit(X_train, y_train)

x_val= vec.transform(val['text'])
y_val= val['emotions'].map(label_map).to_numpy()
y_pred =model.predict(x_val)
accuracy= (y_pred==y_val).mean()
print(f"Validation accuracy: {accuracy:.4f}")  

print(classification_report(
    y_val, y_pred,
    target_names=label_map.keys(), zero_division=0))