#
# Juliane Bruck 
#

ASSIGNMENT #2
==============


BEFORE RUNING:

make sure you have python 3.11.5 installed as well as the following dependencies:

pip install evaluate
pip uninstall urllib3
pip install urllib3


pip install transformers

pip install --upgrade transformers

(you need version 4.39.1 of transformers library see data config function in model_bert.py)




# Required packages
datasets>=2.12.0
pandas>=1.5.3
scikit-learn
transformers>=4.38.2
torch>=2.1.2
accelerate>=0.28.0
# Additional packages
numpy>=1.24.2
scipy>=1.11.3

==========================

This assignment contains 5 python files:

model_bert.py
model_gpt.py
svm_model.py
format_checker.py
scorer.py

No need to worry about the format_checker.py file as it is being called by the scorer.py file.
In order to run the code properly, you sequentially run the following files:


SVM Model
--------
At the command prompt, type: 
> python ./model_svm.py


Uses the following data:
    dev_data = load_data(
        './/data//SubtaskA//medium//subtaskA_dev_monolingual.jsonl')
    train_data = load_data(
        './/data//SubtaskA//medium//subtaskA_train_monolingual.jsonl')
    test_data = load_data('.//data//SubtaskA//medium//subtaskA_monolingual.jsonl')


output:   
     Results_model_svm.jsonl




For results, at the command prompt, type:
> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model_svm.jsonl

INFO : Prediction file format is correct
INFO : macro-F1=0.60538 micro-F1=0.64474        accuracy=0.64474





BERT Model
-----------
At the command prompt, type:
> python ./model_bert.py


Uses the following data:
    dev_data = load_data(
        './/data//SubtaskA//medium//subtaskA_dev_monolingual.jsonl')
    train_data = load_data(
        './/data//SubtaskA//medium//subtaskA_train_monolingual.jsonl')
    test_data = load_data('.//data//SubtaskA//medium//subtaskA_monolingual.jsonl')


output:   
     Results_model_bert.jsonl


For results, at the command prompt, type:
> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model_bert.jsonl
>> 
INFO : Prediction file format is correct
INFO : macro-F1=0.74996 micro-F1=0.75177        accuracy=0.75177


GPT-2 Model 
-----------

> python ./model_gpt.py


Uses the following data:
    dev_data = load_data(
        './/data//SubtaskA//medium//subtaskA_dev_monolingual.jsonl')
    train_data = load_data(
        './/data//SubtaskA//medium//subtaskA_train_monolingual.jsonl')
    test_data = load_data('.//data//SubtaskA//medium//subtaskA_monolingual.jsonl')


output:   
     Results_model_gpt.jsonl


For results, at the command prompt, type:
> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model_gpt.jsonl 
>>
INFO : Prediction file format is correct
INFO : macro-F1=0.36384 micro-F1=0.40789        accuracy=0.40789



     











