# clean tmp
rm -rf tmp

PWD=$(pwd)
# start running
# export TFDS_DATA_DIR='/kmh-nfs-mount/data/tensorflow_datasets'
# export TFDS_DATA_DIR='gs://kmh-gcp/tensorflow_datasets'
# export TFDS_DATA_DIR='gs://kmh-gcp-us-central2/tensorflow_datasets'
python3 main.py \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.cache=True \
    --config.batch_size=1024