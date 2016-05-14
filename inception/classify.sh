#!/bin/bash

set -e

LABELS_FILE=/home/ubuntu/src/tensorflow-test/inception-cars/labels.txt
TF_TEST_DIR=$HOME/src/tensorflow-test
TRAIN_DIR=$TF_TEST_DIR/cars-train/

rm -rf ./tmp
mkdir -p ./tmp/raw/1/
mkdir -p ./tmp/tf/
wget -P ./tmp/raw/1/ $1

bazel-bin/inception/build_image_data \
  --train_directory=./tmp/raw/ \
  --validation_directory=./tmp/raw/ \
  --output_directory=./tmp/tf/ \
  --labels_file="${LABELS_FILE}" \
  --train_shards=1 \
  --validation_shards=1 \
  --num_threads=1

bazel-bin/inception/classify \
  --eval_dir=./tmp/eval/ \
  --data_dir=./tmp/tf/ \
  --subset=validation \
  --num_examples=1 \
  --batch_size=1 \
  --checkpoint_dir="${TRAIN_DIR}" \
  --input_queue_memory_factfor=1 \
  --run_once

