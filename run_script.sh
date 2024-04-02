# clean tmp
[ "$(ls -A ./tmp)" ] && yes | rm -rf './tmp/*' './tmp/*.'

# start running
# export TFDS_DATA_DIR='/kmh-nfs-mount/data/tensorflow_datasets'
# export TFDS_DATA_DIR='gs://kmh-gcp/tensorflow_datasets'
export TFDS_DATA_DIR='gs://kmh-gcp-us-central2/tensorflow_datasets'
python3 main.py \
    --workdir=./tmp --config=configs/tpu.py \
    --config.cache=True