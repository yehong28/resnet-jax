# Run TFDS test in a remote TPU VM

# v2, v3 VMs are deprecated
# VM_NAME=kmh-tpuvm-v3-32-3
# VM_NAME=kmh-tpuvm-v3-128-1
# ZONE=europe-west4-a
# VM_NAME=kmh-tpuvm-v2-32-1
# ZONE=us-central1-a

# VM_NAME=kmh-tpuvm-v4-32
# ZONE=us-central2-b
# SET ENV VARIABLES
echo $VM_NAME $ZONE

CONFIG=tpu
STAGEDIR=/kmh-nfs-ssd-us-mount/code/yehong
LOGDIR=${STAGEDIR}/tmp
# some of the often modified hyperparametes:
batch=1024
PYTHON="/kmh-nfs-ssd-us-mount/code/yehong/miniconda3/envs/pde-tokenization/bin/python"

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR/resnet-jax
echo Current dir: $(pwd)
export MPLCONFIGDIR=mlp_cache
export TFDS_DATA_DIR='gs://kmh-gcp-us-central1/tensorflow_datasets'
$PYTHON tfds_test.py \
        --config=configs/${CONFIG}.py \
        --config.dataset.use_tfds=True \
        --config.dataset.name=imagenet_fake_fake \
        --config.dataset.root='gs://kmh-gcp-us-central1/tensorflow_datasets/' \
        --config.batch_size=${batch} \
        --config.dataset.prefetch_factor=2 \
        --config.dataset.num_workers=32

"
