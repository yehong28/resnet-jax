#!/usr/bin/env bash
set -euo pipefail

# clean tmp
rm -rf tmp


export TFDS_DATA_DIR='gs://kmh-gcp-us-central2/tensorflow_datasets'

PWD=$(pwd)
python3 main.py \
    --debug=False \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.use_tfds=True \
    --config.dataset.name=imagenet_fake \
    --config.dataset.root='gs://kmh-gcp-us-central2/tensorflow_datasets/imagenet_fake' \
    --config.batch_size=1024 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=10 \
    --config.model='_ResNet1'
