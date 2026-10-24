
#clean and prepare the data for training, validation and testing
#we drop examples with multiple emotions and remove duplicates

from pathlib import Path 
import pandas as pd
from datasets import load_dataset

ds= load_dataset("google-research-datasets/go_emotions", "simplified")
names = ds['train'].features['labels'].feature.names
target_emotions = ['joy', 'sadness', 'anger', 'neutral']

def prepare(split):
    df = ds[split].to_pandas()
    df = df[df["labels"].apply(lambda x: len(x) == 1)].copy() #any example with multiple emotions is dropped 
    df["emotions"]= df["labels"].apply(lambda x: names[x[0]])# new col with the emothon name rather than num
    df = df[df["emotions"].isin(target_emotions)].copy()#if it is not one of the 4? drop it
    df["text"] = df["text"].str.lower().str.replace( #rerular cleaning lowercase delete extra spaces etc.. 
       r"\s+", " ",regex=True).str.strip() #regex to replace multiple spaces with a single space and strip leading/trailing spaces
    df = df[df["text"].ne("")] #ne removes empty cols 
    df = df[["text", "emotions"]] #keep only the text and emotions columns

    conflicts= df.groupby("text")["emotions"].transform("nunique")>1 #find examples with(same text different emotions)
    return df[~conflicts].drop_duplicates("text").reset_index(drop=True) #drop them and reset the index



train= prepare("train")
val = prepare("validation")
test = prepare("test")
#droping examples from val and test if they are also in train (preventing leakage )
val = val[~val["text"].isin(train["text"])]
test = test[~test["text"].isin(pd.concat([train["text"], val["text"]]))]

#we saved the clean data into 3 csv files on my laptop
Path("data").mkdir(exist_ok=True)
for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(f"data/{name}.csv", index=False)
        print(f"\n{name}: {len(df)} examples")
        print(df["emotions"].value_counts())