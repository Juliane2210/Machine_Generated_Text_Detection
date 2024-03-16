#
# baseline model SVM model
#

import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

# Step 1: Data Loading


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)


# minimized data for development
dev_data = load_data(
    './/data//SubtaskA//min//subtaskA_dev_monolingual.jsonl')
train_data = load_data(
    './/data//SubtaskA//min//subtaskA_train_monolingual.jsonl')
test_data = load_data('.//data//SubtaskA//min//subtaskA_monolingual.jsonl')

# final data for proper results
# dev_data = load_data('.//data//SubtaskA//subtaskA_dev_monolingual.jsonl')
# train_data = load_data('.//data//SubtaskA//subtaskA_train_monolingual.jsonl')
# test_data = load_data('.//data//SubtaskA//subtaskA_monolingual.jsonl')

# Step 2: Data Preprocessing


def preprocess_text(text):
    # Remove special characters, punctuation, and extra whitespaces
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Tokenization (split text into words)
    tokens = text.split()
    # Convert tokens to lowercase
    tokens = [token.lower() for token in tokens]
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


train_data['text'] = train_data['text'].apply(preprocess_text)
dev_data['text'] = dev_data['text'].apply(preprocess_text)
test_data['text'] = test_data['text'].apply(preprocess_text)

# Step 3: Feature Engineering
# For this example, we'll use TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']
# Transform dev data using the same vectorizer
X_dev = vectorizer.transform(dev_data['text'])
y_dev = dev_data['label']

# Step 4: Model Training
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the trained model
model_filename = 'svm_model.joblib'
dump(model, model_filename)

# Step 5: Model Evaluation
dev_predictions = model.predict(X_dev)
dev_accuracy = accuracy_score(y_dev, dev_predictions)
dev_f1_macro = f1_score(
    y_dev, dev_predictions, average='macro')
dev_f1_micro = f1_score(
    y_dev, dev_predictions, average='micro')
print("Dev Accuracy:", dev_accuracy)
print("Dev Macro F1:", dev_f1_macro)
print("Dev Micro F1:", dev_f1_micro)


# Step 6: Prediction
X_test = vectorizer.transform(test_data['text'])
test_predictions = model.predict(X_test)

# Predictions on dev set
y_test = test_data['id']
dev_accuracy = accuracy_score(y_test, test_predictions)

test_f1_macro = f1_score(
    y_test, test_predictions, average='macro')
test_f1_micro = f1_score(
    y_test, test_predictions, average='micro')
print("Test Accuracy:", test_predictions)
print("Test Macro F1:", test_f1_macro)
print("Test Micro F1:", test_f1_micro)


# Step 7: Output Generation
output_file = 'Results_model1.jsonl'
with open(output_file, 'w') as file:
    for idx, label in zip(test_data['id'], test_predictions):
        file.write(json.dumps({'id': idx, 'label': int(label)}) + '\n')


# Step 8: Validate Results.jsonl
#  python .\format_checker.py --pred_files_path .\Results.jsonl
#

# Step 9: Scoring
# Use the provided scorer script to compute scores
# python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model1.jsonl

# Step 10: Report
# Write a report summarizing your approach, models used, evaluation results, etc.
