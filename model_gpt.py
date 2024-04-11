#
# CSI 5386 Assignment 2
# Juliane Bruck 8297746
#
# GPT2 model
#

from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import logging
import json
import evaluate
import re


#
# Cleaning step.  Note, in the baseline, truncation was being used.
#

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def get_data(train_path, test_path, random_seed):
    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    f1_metric = evaluate.load("f1")
    results = f1_metric.compute(
        predictions=predictions, references=labels, average="weighted")

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model_name):
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    configuration = GPT2Config()
    GPT2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    GPT2_tokenizer.pad_token = GPT2_tokenizer.eos_token
    model = GPT2ForSequenceClassification(
        configuration).from_pretrained(model_name)
    model.config.pad_token_id = model.config.eos_token_id

    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, GPT2_tokenizer), batched=True)
    tokenized_valid_dataset = valid_dataset.map(
        lambda examples: preprocess_function(examples, GPT2_tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=GPT2_tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=GPT2_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_model_path = os.path.join(checkpoints_path, 'best')
    os.makedirs(best_model_path, exist_ok=True)
    trainer.save_model(best_model_path)


def test(test_df, model_path, id2label, label2id):
    GPT2_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    GPT2_tokenizer.pad_token = GPT2_tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(
        model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    test_dataset = Dataset.from_pandas(test_df)
    tokenized_test_dataset = test_dataset.map(
        lambda examples: preprocess_function(examples, GPT2_tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=GPT2_tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=GPT2_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)

    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(
        predictions=preds, references=predictions.label_ids)

    return results, preds


if __name__ == '__main__':
    random_seed = 0
    train_path = './data/SubtaskA/medium/subtaskA_train_monolingual.jsonl'
    test_path = './data/SubtaskA/medium/subtaskA_monolingual_gold.jsonl'
    model = 'gpt2'
    prediction_path = "./Results_model_gpt.jsonl"
    subtask = "A"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or test file not found")

    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}

    set_seed(random_seed)

    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    fine_tune(train_df, valid_df,
              f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)

    results, predictions = test(
        test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)

    print(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')


# Step 8: Validate Results.jsonl
#  python .\format_checker.py --pred_files_path .\Results.jsonl
#

# Step 9: Scoring
# Use the provided scorer script to compute scores
# python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model_gpt.jsonl

# Step 10: Report
# Write a report summarizing your approach, models used, evaluation results, etc.
