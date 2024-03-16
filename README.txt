

Model 1
--------

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


PS D:\juliane\CSI5180\assignment5> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results.jsonl
INFO : Prediction file format is correct
INFO : macro-F1=0.51967 micro-F1=0.57263        accuracy=0.57263





Model 2
--------

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

> python ./baseline.py


Uses the following data:
    dev_data = load_data(
        './/data//SubtaskA//min//subtaskA_dev_monolingual.jsonl')
    train_data = load_data(
        './/data//SubtaskA//min//subtaskA_train_monolingual.jsonl')
    test_data = load_data('.//data//SubtaskA//min//subtaskA_monolingual.jsonl')


output:   
     Results_baseline1.jsonl







------------------------------------- RAW NOTES -------------------







application.py
----------------


output:    Results.jsonl


Dev Accuracy: 0.5315474092351075
Dev Macro F1: 0.4890709260655781
Dev Micro F1: 0.5315474092351075
Test Accuracy: [0 0 1 ... 0 0 0]
Test Macro F1: 3.572305668156863e-09
Test Micro F1: 3.914353935882883e-05


PS D:\juliane\CSI5180\assignment5> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results.jsonl
INFO : Prediction file format is correct
INFO : macro-F1=0.51967 micro-F1=0.57263        accuracy=0.57263







wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz





PS D:\juliane\CSI5180\assignment5> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_bert.jsonl
INFO : Prediction file format is correct
INFO : macro-F1=0.32208 micro-F1=0.47511        accuracy=0.47511
PS D:\juliane\CSI5180\assignment5> 



Test Macro F1: 2.34000234000234e-06
Test Micro F1: 0.0010822510822510823
PS D:\juliane\CSI5180\assignment5> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_bert.jsonl
INFO : Prediction file format is correct
INFO : macro-F1=0.32208 micro-F1=0.47511        accuracy=0.47511
PS D:\juliane\CSI5180\assignment5> 






https://huggingface.co/google/mt5-small

google/t5-efficient-tiny-dl2



openai-community/gpt2





https://huggingface.co/google/bigbird-roberta-large







model1---------------------------------

PS D:\juliane\CSI5180\assignment5> python .\model1.py
Dev Accuracy: 0.642512077294686
Dev Macro F1: 0.5815668706293706
Dev Micro F1: 0.642512077294686
Test Accuracy: [1 1 1 1 1 0 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1
 1 1 1 0 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 0 1 1 0 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 0 1 0 0 1 1 1 1 1 1]
Test Macro F1: 0.00012334258402713538
Test Micro F1: 0.0070921985815602835
PS D:\juliane\CSI5180\assignment5> 



PS D:\juliane\CSI5180\assignment5> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model1.jsonl
INFO : Prediction file format is correct
INFO : macro-F1=0.69348 micro-F1=0.73050        accuracy=0.73050
PS D:\juliane\CSI5180\assignment5> 


-------------------------------------------------------------
MODEL2


PS D:\juliane\CSI5180\assignment5> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model2.jsonl
INFO : Prediction file format is correct
INFO : macro-F1=0.53083 micro-F1=0.57447        accuracy=0.57447
PS D:\juliane\CSI5180\assignment5> 



---------------------------------------------------------------



MODEL5



PS D:\juliane\CSI5180\assignment5> python .\model5.py
2024-03-16 09:47:29.396874: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\bruck\AppData\Roaming\Python\Python311\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

Some weights of BertForSequenceClassification were not initialized from the model checkpoint at google-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
C:\Users\bruck\anaconda3\Lib\site-packages\accelerate\accelerator.py:432: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches', 'even_batches', 'use_seedable_sampler']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False, even_batches=True, use_seedable_sampler=True)
  warnings.warn(
{'eval_loss': 0.6614343523979187, 'eval_f1': 0.4444444444444444, 'eval_runtime': 206.9076, 'eval_samples_per_second': 1.0, 'eval_steps_per_second': 0.019, 'epoch': 1.0}                                                                                                                                  
{'eval_loss': 0.5926603078842163, 'eval_f1': 0.5893719806763285, 'eval_runtime': 208.513, 'eval_samples_per_second': 0.993, 'eval_steps_per_second': 0.019, 'epoch': 2.0}
{'train_runtime': 3037.941, 'train_samples_per_second': 0.143, 'train_steps_per_second': 0.003, 'train_loss': 0.6283450126647949, 'epoch': 2.0}      
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [50:37<00:00, 379.74s/it] 
C:\Users\bruck\anaconda3\Lib\site-packages\accelerate\accelerator.py:432: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches', 'even_batches', 'use_seedable_sampler']). Please pass an `accelerate.DataLoaderConfiguration` instead:
dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False, even_batches=True, use_seedable_sampler=True)
  warnings.warn(
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18/18 [02:17<00:00,  7.65s/it] 
PS D:\juliane\CSI5180\assignment5> python scorer.py --gold_file_path .\data\SubtaskA\subtaskA_monolingual_gold.jsonl --pred_file_path .\Results_model5.jsonl
INFO : Prediction file format is correct
INFO : macro-F1=0.43838 micro-F1=0.51773        accuracy=0.51773

