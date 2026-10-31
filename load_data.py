
#just quick look on the data we want to use

from collections import Counter

from datasets import load_dataset

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")

print(dataset)
print("\nFirst example:")
print(dataset['train'][0])

label_names = dataset['train'].features['labels'].feature.names
target_emotions = ['joy', 'sadness', 'anger', 'neutral']

# Print the distribution of emotions in the training set
emotion_counts = Counter()
for example in dataset['train']:
    if len(example["labels"]) == 1:
        label_id = example["labels"][0]
        emotion = label_names[label_id]

        if emotion in target_emotions:
            emotion_counts[emotion] += 1


print("\nEmotion distribution:")
for emotion in target_emotions:
    print(f"{emotion}: {emotion_counts[emotion]}")
