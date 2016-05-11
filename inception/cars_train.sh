#!/bin/bash

TF_TEST_DIR=$HOME/src/tensorflow-test
TRAIN_DIR=$TF_TEST_DIR/cars-train/
CARS_DATA_DIR=$TF_TEST_DIR/cars-data/
MODEL_PATH=$TF_TEST_DIR/inception-v3/model.ckpt-157585

bazel-bin/inception/cars_train \
  --train_dir="${TRAIN_DIR}" \
  --data_dir="${CARS_DATA_DIR}" \
  --pretrained_model_checkpoint_path="${MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.01 \
  --input_queue_memory_factor=1 \
  --num_gpus=4 \
  --batch_size=100
