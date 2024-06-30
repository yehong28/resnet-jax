# clean tmp
rm -rf tmp

# activate conda env
conda activate resnet_jax

PWD=$(pwd)
python3 main.py \
    --debug=False \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.root=./imagenet_fake \
    --config.batch_size=1024 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=10 \
    --config.model='_ResNet1'