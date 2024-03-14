

Model 1
--------
Run: 

> python ./application.py


Uses the following data:
    dev_data = load_data(
        './/data//SubtaskA//min//subtaskA_dev_monolingual.jsonl')
    train_data = load_data(
        './/data//SubtaskA//min//subtaskA_train_monolingual.jsonl')
    test_data = load_data('.//data//SubtaskA//min//subtaskA_monolingual.jsonl')


output:   
     Results.jsonl


Dev Accuracy: 0.5315474092351075
Dev Macro F1: 0.4890709260655781
Dev Micro F1: 0.5315474092351075
Test Accuracy: [0 0 1 ... 0 0 0]
Test Macro F1: 3.572305668156863e-09
Test Micro F1: 3.914353935882883e-05

Run:
> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results.jsonl

Output:
INFO : Prediction file format is correct
INFO : macro-F1=0.51967 micro-F1=0.57263        accuracy=0.57263





Model 2
--------

Run:

> python ./bertModel.py


Uses the following data:
    dev_data = load_data(
        './/data//SubtaskA//min//subtaskA_dev_monolingual.jsonl')
    train_data = load_data(
        './/data//SubtaskA//min//subtaskA_train_monolingual.jsonl')
    test_data = load_data('.//data//SubtaskA//min//subtaskA_monolingual.jsonl')


output:   
     Results_bert.jsonl





Model 3
-------


Run:
> python ./baseline.py


Uses the following data:
    dev_data = load_data(
        './/data//SubtaskA//min//subtaskA_dev_monolingual.jsonl')
    train_data = load_data(
        './/data//SubtaskA//min//subtaskA_train_monolingual.jsonl')
    test_data = load_data('.//data//SubtaskA//min//subtaskA_monolingual.jsonl')


output:   
     Results_baseline1.jsonl