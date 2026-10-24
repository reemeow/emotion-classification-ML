import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from LR import LogisticRegressionScrtach
from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")


emotion_counts = train['emotions'].value_counts()
plt.figure(figsize=(7, 4))
emotion_counts.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen', 'orange'])
plt.title('Distribution of Emotions in Training Set')
plt.xlabel('Emotions')
plt.ylabel('Number of Samples')
plt.xticks(rotation=0)
plt.savefig('emotion_distribution.png')
plt.show()

vec = CountVectorizer(min_df=2, max_features=5000)
X_train = vec.fit_transform(train['text'])

label_map = {"joy": 0, "sadness": 1, "anger": 2, "neutral": 3}
y_train = train['emotions'].map(label_map).to_numpy()
counts =np.bincount(y_train)
print("class coounts:", counts)
clas_weights=len(y_train)/ (4*counts)
print("class weights:", clas_weights)  

model = LogisticRegressionScrtach(
    n_features=X_train.shape[1],
    n_classes=4,
    class_weights=clas_weights
)
print("\n--- Starting Training ---")
model.fit(X_train, y_train, learning_rate=1.0, epochs=500) 


x_val= vec.transform(val['text'])
y_val= val['emotions'].map(label_map).to_numpy()
y_pred =model.predict(x_val)
accuracy= (y_pred==y_val).mean()
print(f"Validation accuracy: {accuracy:.4f}")  

cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3])

print("Confusion Matrix:")
print(cm)

print(classification_report(
    y_val, y_pred,
    target_names=label_map.keys(), zero_division=0))