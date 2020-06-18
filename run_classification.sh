#!/bin/bash
# Run this before if needed : sed -i -e 's/\r$//' run_classification.sh
# Finally, run this command : sh run_classification.sh

mkdir -p 5-fold-hyper-tune

for MODEL in books_to_dvd
do
  SRC_DOMAIN="${MODEL%_to_*}" # split model name according to '_to_' and take the prefix
  TRG_DOMAIN="${MODEL#*_to_}" # split model name according to '_to_' and take the suffix
  mkdir 5-fold-hyper-tune/${MODEL}/

  TEMP_DIR=models/${MODEL}/temp
  mkdir -p ${TEMP_DIR}/
  mkdir -p 5-fold-hyper-tune/${MODEL}/

  MODELS_DIR=models/${MODEL}/

  for EPOCH in 1             # for hyper-tuning run over [20 40 60]
  do
    for OUT_CHANNELS in 32    # for hyper-tuning run over [16 32 64]
    do
      for BATCH_SIZE in 32    # for hyper-tuning run over [32 64]
      do
        for FILTER_SIZE in 9  # for hyper-tuning run over [7 9 11]
        do
          for FOLD_NUM in 1   # for five-fold run over [1 2 3 4 5]
          do
            cp ${MODELS_DIR}pytorch_model${EPOCH}.bin ${TEMP_DIR}

            python supervised_task_learning.py \
            --in_domain_data_dir=data/${SRC_DOMAIN}/ \
            --cross_domain_data_dir=data/${TRG_DOMAIN}/ \
            --do_train \
            --output_dir=${TEMP_DIR}/ \
            --load_model \
            --model_name=pytorch_model${EPOCH}.bin \
            --cnn_window_size=${FILTER_SIZE} \
            --cnn_out_channels=${OUT_CHANNELS} \
            --learning_rate=5e-5 \
            --train_batch_size=${BATCH_SIZE} \
            --use_fold=True \
            --fold_num=${FOLD_NUM} \
            --save_according_to=loss

            COPY_FROM_PATH=${TEMP_DIR}/pytorch_model${EPOCH}.bin-final_eval_results.txt

            COPY_TO_PATH=5-fold-hyper-tune/${MODEL}/ep-${EPOCH}_ch-${OUT_CHANNELS}_batch-${BATCH_SIZE}_filt-${FILTER_SIZE}_fold-${FOLD_NUM}.txt
            cp ${COPY_FROM_PATH} ${COPY_TO_PATH}
            rm ${TEMP_DIR}/*
              done
            done
          done
        done
      done
    done
  done
done