# clean tmp
rm -rf tmp

PWD=$(pwd)
python3 main.py \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.batch_size=1024