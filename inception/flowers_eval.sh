#!/bin/bash

TF_TEST_DIR=$HOME/src/tensorflow-test
TRAIN_DIR=$TF_TEST_DIR/flowers-train/
FLOWERS_DATA_DIR=$TF_TEST_DIR/flowers-data/
EVAL_DIR=$TF_TEST_DIR/flowers-eval/

bazel-bin/inception/flowers_eval \
  --eval_dir="${EVAL_DIR}" \
  --data_dir="${FLOWERS_DATA_DIR}" \
  --subset=validation \
  --num_examples=500 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factfor=1 \
  --run_once

