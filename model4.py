import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
import os
import logging
import json

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


def preprocess_function(examples):
    # Tokenize the input text sequences
    return tokenizer(examples["text"], truncation=True)


def get_data(train_path, valid_path, test_path,  random_seed):
    # Read the train and test data from JSON files
    train_df = pd.read_json(train_path, lines=True)
    valid_df = pd.read_json(valid_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    return train_df, valid_df, test_df


def compute_metrics(eval_pred):
    # Define your metrics computation here
    pass


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):
    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # Tokenize the training and validation datasets
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, batched=True)
    tokenized_valid_dataset = valid_dataset.map(
        preprocess_function, batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the best model
    trainer.save_model(checkpoints_path)


def test(test_df, model_path, id2label, label2id):
    # Load the trained model
    model = GPT2ForSequenceClassification.from_pretrained(model_path)

    test_dataset = Dataset.from_pandas(test_df)
    # Tokenize the test dataset
    tokenized_test_dataset = test_dataset.map(
        preprocess_function, batched=True)

    # Initialize the Trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

 # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Evaluate the model on the test dataset
    results = trainer.evaluate(tokenized_test_dataset)

    return results, preds


if __name__ == '__main__':
    # Set up logging and other configurations

    # Define paths and other configurations
    random_seed = 0
    train_path = './/data//SubtaskA//min1//subtaskA_train_monolingual.jsonl'
    test_path = './/data//SubtaskA//min1//subtaskA_monolingual.jsonl'
    valid_path = './/data//SubtaskA//min1//subtaskA_dev_monolingual.jsonl'
    model = 'openai-community/gpt2'
    prediction_path = ".//Results_model4_1.jsonl"
    subtask = "A"
    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere',
                    3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1, 'cohere': 2,
                    'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error(
            "Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError(
            "Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)
    # Load GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model)
    model = GPT2ForSequenceClassification.from_pretrained(model)

    # Get train and test data
    train_df, valid_df, test_df = get_data(
        train_path, valid_path, test_path, random_seed)

    # Fine-tune the model
    fine_tune(train_df, valid_df,
              f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)

    # Test the model
    results, predictions = test(
        test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)

    # Output results and other relevant information

    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')

    # Step 7: Output Generation
    output_file = 'Results_model4.jsonl'
    with open(output_file, 'w') as file:
        for idx, label in zip(test_df['id'], predictions):
            file.write(json.dumps({'id': idx, 'label': int(label)}) + '\n')


# Step 8: Validate Results.jsonl
#  python .\format_checker.py --pred_files_path .\Results.jsonl
#

# Step 9: Scoring
# Use the provided scorer script to compute scores
# python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model3.jsonl

# Step 10: Report
# Write a report summarizing your approach, models used, evaluation results, etc.
