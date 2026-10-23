from pathlib import Path 
import pandas as pd
from datasets import load_dataset

ds= load_dataset("google-research-datasets/go_emotions", "simplified")
names = ds['train'].features['labels'].feature.names
target_emotions = ['joy', 'sadness', 'anger', 'neutral']

def prepare(split):
    df = ds[split].to_pandas()
    df = df[df["labels"].apply(lambda x: len(x) == 1)].copy()
    df["emotions"]= df["labels"].apply(lambda x: names[x[0]])
    df = df[df["emotions"].isin(target_emotions)].copy()
    df["text"] = df["text"].str.lower().str.replace(
       r"\s+", " ",regex=True).str.strip()
    df = df[df["text"].ne("")]
    df = df[["text", "emotions"]]

    conflicts= df.groupby("text")["emotions"].transform("nunique")>1
    return df[~conflicts].drop_duplicates("text").reset_index(drop=True)



train= prepare("train")
val = prepare("validation")
test = prepare("test")

val = val[~val["text"].isin(train["text"])]
test = test[~test["text"].isin(pd.concat([train["text"], val["text"]]))]

Path("data").mkdir(exist_ok=True)

for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(f"data/{name}.csv", index=False)
        print(f"\n{name}: {len(df)} examples")
        print(df["emotions"].value_counts())