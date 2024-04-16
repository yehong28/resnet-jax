# clean tmp
rm -rf tmp

PWD=$(pwd)
python3 main.py \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.batch_size=256 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=8 \
    --config.dataset.root=/home/kaiminghe/data_local/imagenet_fake \
    --config.log_per_step=10 \
    --config.model='ResNet50'
