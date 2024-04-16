# clean tmp
rm -rf tmp

PWD=$(pwd)
python3 main.py \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.batch_size=128 \
    --config.dataset.num_workers=32 \
    --config.dataset.root=imagenet_fake \
    --config.model='_ResNet1'
