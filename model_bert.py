#
# CSI 5386 Assignment 2
# Juliane Bruck 8297746
#
#
# Bert model uncased
#
import re
from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, set_seed
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import json
from accelerate import DataLoaderConfiguration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#
# Cleaning step
#
def clean_sentence(sentence):
    # Remove special characters, punctuation, and numbers
    cleaned_sentence = re.sub(r'[^A-Za-z\s]', '', sentence)
    # Convert to lowercase
    cleaned_sentence = cleaned_sentence.lower()
    # Remove extra whitespaces
    cleaned_sentence = ' '.join(cleaned_sentence.split())

    return cleaned_sentence


def preprocess_function(examples, **fn_kwargs):
    cleaned_text = [clean_sentence(text) for text in examples["text"]]
    return fn_kwargs['tokenizer'](cleaned_text, truncation=True, padding=True)


def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    train_df, val_df = train_test_split(
        train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    return train_df, val_df, test_df


def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions,
                   references=labels, average="weighted"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):

    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    model = AutoModelForSequenceClassification.from_pretrained(
        # put your model here
        model, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(
        preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#Older version of Transformers (4.32.1) requires a dataloaderconfig in the trainer
    # Define your DataLoaderConfiguration
    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=None,
        split_batches=False,
        even_batches=True,
        use_seedable_sampler=True
    )

    # create Trainer
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
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
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        #dataloader_config=dataloader_config #see data loader config definition 
    )

    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    trainer.save_model(best_model_path)


def test(test_df, model_path, id2label, label2id):

    # load tokenizer from saved model
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(
        preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(
        predictions=preds, references=predictions.label_ids)

    print(results)
    # return dictionary of classification report
    return preds


if __name__ == '__main__':

    random_seed = 0
    train_path = './/data//SubtaskA//medium//subtaskA_train_monolingual.jsonl'
    test_path = './/data//SubtaskA//medium//subtaskA_monolingual.jsonl'
    model = 'google-bert/bert-base-uncased'
    prediction_path = ".//Results_model_bert.jsonl"
    subtask = "A"

    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))

    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}

    set_seed(random_seed)

    # get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)

    # train detector model
    fine_tune(train_df, valid_df,
              f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)

    # test detector model
    predictions = test(
        test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)

    # logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')


# Step 8: Validate Results.jsonl
#  python .\format_checker.py --pred_files_path .\Results.jsonl
#

# Step 9: Scoring
# Use the provided scorer script to compute scores
# python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model_bert.jsonl

# Step 10: Report
# Write a report summarizing your approach, models used, evaluation results, etc.
