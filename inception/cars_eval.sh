#!/bin/bash

TF_TEST_DIR=$HOME/src/tensorflow-test
TRAIN_DIR=$TF_TEST_DIR/cars-train/
CARS_DATA_DIR=$TF_TEST_DIR/cars-data/
EVAL_DIR=$TF_TEST_DIR/cars-eval/

bazel-bin/inception/cars_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${CARS_DATA_DIR}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factfor=1 \
  --run_once

