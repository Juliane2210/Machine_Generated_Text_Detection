import json
import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

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

# Step 3: Tokenization and Data Preparation for BERT


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, 'text'])
        label = int(self.data.loc[idx, 'label'])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128

train_dataset = CustomDataset(train_data, tokenizer, MAX_LEN)
dev_dataset = CustomDataset(dev_data, tokenizer, MAX_LEN)
test_dataset = CustomDataset(test_data, tokenizer, MAX_LEN)

# Step 4: Model Training
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

epochs = 3

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Step 5: Model Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in dev_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)

        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

dev_accuracy = accuracy_score(true_labels, predictions)
dev_f1_macro = f1_score(true_labels, predictions, average='macro')
dev_f1_micro = f1_score(true_labels, predictions, average='micro')
print("Dev Accuracy:", dev_accuracy)
print("Dev Macro F1:", dev_f1_macro)
print("Dev Micro F1:", dev_f1_micro)

# Step 6: Prediction
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)

        predictions.extend(predicted_labels.cpu().numpy())

# Step 7: Output Generation
output_file = 'Results_bert.jsonl'
with open(output_file, 'w') as file:
    for idx, label in zip(test_data['id'], predictions):
        file.write(json.dumps({'id': idx, 'label': int(label)}) + '\n')
