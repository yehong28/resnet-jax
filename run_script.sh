# clean tmp
rm -rf tmp

PWD=$(pwd)
python3 main.py \
    --debug=True \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.root=./imagenet_fake \
    --config.batch_size=4096 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=10 \
    --config.model='_ResNet1'