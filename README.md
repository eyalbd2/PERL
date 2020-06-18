# PERL
Official code for our TACL'20 paper "PERL: Pivot-based Domain Adaptation for Pre-trained Deep Contextualized Embedding Models" - [https://arxiv.org/abs/2006.09075](https://arxiv.org/abs/2006.09075). 


If you use this code please cite our paper.
 
PERL is a pivot-based domain adaptation model for text-classification. 
The model adjusts representations from massively pre-trained contextualized 
word embedding models such as BERT so that they close the gap between the 
source and target domains. This is achieved by finetune their parameters 
using a pivot-based variant of the Masked Language Modeling (MLM) objective, 
optimized on unlabeled data from both the source and the target domains.


PERL is built on Python 3, pytorch and  
[pytorch Pre-trained BERT](https://github.com/huggingface/pytorch-pretrained-BERT). 
It works with CPU and GPU.

## Usage Instructions

Running an experiment with PERL consists of the following steps (for each pair of domains):

1. Select pivot features. sing mutual information criteria and appearance frequency.
2. Run pivot-based finetuning on a pre-trained BERT-base-cased. We use a pivot-based variant of the MLM objective, on unlabeled data from the source and target domain.
3. Use finetuned encoder and train a classifier on source domain labeled data.
4. Use PERL to predict sentiment on target domain data.
5. Evaluate the predictions.

Next we go through these stepson the 'books_to_dvd' pair as a
running example. We use a specific set of hyperparamers, and run only a one-fold experiment instead of 5 (please read the paper for moer details). 

You can run all of the steps with

```
sh run_full_experiment.sh
```
### 0. Setup a virtual env
Make sure you meet all requirements in 'requirements.txt'.


after setting the paths in the beginning of the script.

### 1. Select pivot features

Make sure you download both our [data directory](https://github.com/eyalbd2/PERL/tree/master/data)
and our [5-fold-data directory](https://github.com/eyalbd2/PERL/tree/master/5-fold_data). Then run the following command to find a set the appropriate set of pivot features.

```
This will save a file named '100_bi' in the directoryMODEL=books_to_dvd
SRC_DOMAIN="${MODEL%_to_*}" # split model name according to '_to_' and take the prefix
TRG_DOMAIN="${MODEL#*_to_}" # split model name according to '_to_' and take the suffix

NUM_PIVOTS=100
PIV_MN_ST=20

python utils/pivot_selection.py \
--pivot_num=${NUM_PIVOTS} \
--pivot_min_st=${PIV_MN_ST} \
--src=${SRC_DOMAIN} \
--dest=${TRG_DOMAIN}
```

This will save a file named '100_bi' in the path - 'data/pivots/books_to_dvd/'.

### 2. Run pivot-based finetuning on a pre-trained BERT-base-cased

```
PIVOT_PROB=0.5
NON_PIVOT_PROB=0.1
NUM_PRE_TRAIN_EPOCHS=20
SAVE_FREQ=10
UNFROZEN_BERT_LAYERS=8

mkdir -p models/${MODEL}

OUTPUT_DIR_NAME=models/${MODEL}
PIVOTS_PATH=data/pivots/${MODEL}/100_bi

python perl_pretrain.py \
 --src_domain=${SRC_DOMAIN} \
 --trg_domain=${TRG_DOMAIN} \
 --pivot_path=${PIVOTS_PATH} \
 --output_dir=${OUTPUT_DIR_NAME} \
 --num_train_epochs=${NUM_PRE_TRAIN_EPOCHS} \
 --save_every_num_epochs=${SAVE_FREQ} \
 --pivot_prob=${PIVOT_PROB} \
 --non_pivot_prob=${NON_PIVOT_PROB} \
 --num_of_unfrozen_bert_layers=${UNFROZEN_BERT_LAYERS} \
 --init_output_embeds \
 --train_output_embeds
```
This code creates a dir named 'model/books_to_dvd/' and saves finetuned models: 'pytorch_model10.bin' 'pytorch_model20.bin' and 'pytorch_model.bin' - one at the end of finetuning stage, and the others are after each 'SAVE_FREQ' epochs of finetuning. 
** model hyperparameters grid for this step are specified in the paper.

### 3. Train a classifier on source domain labeled data the predict and evaluate on target domain

Train the classifier on labeld data from the source domain, while using the finetuned encoder.  
You can specify which finetuned checkpoint to use by setting 'PRE_TRAINED_EPOCH' variable.

First, create the results directory '5-fold-hyper-tune/books_to_dvd/'
```
mkdir -p 5-fold-hyper-tune
mkdir -p 5-fold-hyper-tune/${MODEL}/
```
Next create temp dir in the model directory to save the results. 
```
TEMP_DIR=models/${MODEL}/temp
mkdir -p ${TEMP_DIR}/
```
Set hyperparams and train the model.
```
PRE_TRAINED_EPOCH=20
CNN_OUT_CHANNELS=32
BATCH_SIZE=32
CNN_FILTER_SIZE=9
FOLD_NUM=1

python run_lasertagger.py \
  --training_file=${OUTPUT_DIR}/train.tf_record \
  --eval_file=${OUTPUT_DIR}/tune.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --do_train=true \
  --do_eval=true \
  --train_batch_size=256 \
  --save_checkpoints_steps=500 \
  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
  --num_eval_examples=${NUM_EVAL_EXAMPLES}
```

Example output at the end of each epoch:
```
acc: 0.9756048387096774
loss: 0.11142500407993794
dev_acc: 0.8421052631578947
dev_loss: 0.34983451664447784
dev_cross_acc: 0.8614307153576788
dev_cross_loss: 0.317698473110795
06/18/2020 07:16:04 - INFO - __main__ -   ***** Evaluation results *****
06/18/2020 07:16:04 - INFO - __main__ -     acc = 0.9756048387096774
06/18/2020 07:16:04 - INFO - __main__ -     dev_acc = 0.8421052631578947
06/18/2020 07:16:04 - INFO - __main__ -     dev_cross_acc = 0.8614307153576788
06/18/2020 07:16:04 - INFO - __main__ -     dev_cross_loss = 0.317698473110795
06/18/2020 07:16:04 - INFO - __main__ -     dev_loss = 0.34983451664447784
06/18/2020 07:16:04 - INFO - __main__ -     loss = 0.11142500407993794
Best results in domain: Loss - 0.34971608434404644, Cross Acc - 0.8624312156078039
```
The last line presents the cross-domain accuracy of the model that achieves the best score according to the in-domain development set (using either accuracy or loss criteria). Note that after the last epoch this line reports the final result of the model.

### Full hyperparameter tuning
To tune the hyperparameters of the model, we use a five-fold cross validation protocol. We calculate the mean score across all five folds and choose the set of hyperparameters that achieves the lowest loss value according to developement set from the source domain.
To read more about the full hyperparameter grid, see [full paper](https://arxiv.org/abs/2006.09075)

### More running possibilities
You can ran each step seperately (finetuning, classification) for many setups at once using the following files:
- 'run_perl_pretraining.sh' - for finetuning, specify which setups you want to experiment in the line 'for MODEL in [many setups seperated by space]' 
- 'run_classification.sh' - for classification, specify hyperparameters.

All results of the second step here will be saved under '5-fold-hyper-tune'. You need to calculate mean across folds for each set of hyperparameters and choose the best set.

## (TODO) How to Cite PERL

```
@inproceedings{
}
```
