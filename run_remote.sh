# Run job in a remote TPU VM
VM_NAME=kmh-tpuvm-v3-32-4
ZONE=europe-west4-a  # v3

CONFIG=tpu

# some of the often modified hyperparametes:
batch=4096
lr=0.1
ep=100

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=resnet/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_ep${ep}_torchvision

LOGDIR=/kmh-nfs-mount/logs/$USER/$JOBNAME
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

echo 'Log dir: '$LOGDIR

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
echo Current dir: $(pwd)

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root='/kmh-nfs-mount/data/imagenet' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=64 \
    --config.log_per_step=20 \
    --config.model='ResNet50'
" 2>&1 | tee -a $LOGDIR/output.log

    # --config.dataset.root='/kmh-nfs-mount/data/imagenet' \
