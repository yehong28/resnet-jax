# clean tmp
rm -rf tmp

PWD=$(pwd)
python3 main.py \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.root=./imagenet_fake \
    --config.batch_size=1024 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=10 \
    --config.model='_ResNet1'

    # --config.dataset.root=/kmh-nfs-us-mount/data/imagenet \
