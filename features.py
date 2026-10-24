#an extra file just to show how we transform examples into vectors 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("data/train.csv")

vec = CountVectorizer(
    min_df=2, max_features=5000
)

x= vec.fit_transform(train["text"]) 
label_map={"joy":0, "sadness":1, "anger":2, "neutral":3}
y= train["emotions"].map(label_map).to_numpy() 

print("Num of sentences: ", x.shape[0])
print("Num of features: ", x.shape[1])

print("\nFirst sentences:")
print(train["text"].iloc[0])

print("\nits vector: ")
words= vec.get_feature_names_out()
indices= x[0].indices
counts= x[0].data
for index, count in zip(indices, counts):
    print(words[index], ";", count)

print("\n First 20 words in vocab:")
print(vec.get_feature_names_out()[:20])
      
print("y shape", y.shape)
print("Labels:", np.unique(y, return_counts=True))