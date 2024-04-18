# Run job in a remote TPU VM

# VM_NAME=kmh-tpuvm-v3-32-3
VM_NAME=kmh-tpuvm-v3-128-1
ZONE=europe-west4-a

# VM_NAME=kmh-tpuvm-v4-32
# ZONE=us-central2-b

# VM_NAME=kmh-tpuvm-v2-32-1
# ZONE=us-central1-a

echo $VM_NAME $ZONE

CONFIG=tpu

# some of the often modified hyperparametes:
batch=16384
lr=0.1
ep=100

now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
JOBNAME=resnet/${now}_${salt}_${VM_NAME}_${CONFIG}_b${batch}_lr${lr}_ep${ep}_torchvision

LOGDIR=/kmh-nfs-ssd-eu-mount/logs/$USER/$JOBNAME
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

echo 'Log dir: '$LOGDIR

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
echo Current dir: $(pwd)

python3 main.py \
    --workdir=${LOGDIR} --config=configs/${CONFIG}.py \
    --config.dataset.root='/kmh-nfs-ssd-eu-mount/data/imagenet' \
    --config.batch_size=${batch} \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=16 \
    --config.log_per_step=20 \
    --config.model='ResNet152'
" 2>&1 | tee -a $LOGDIR/output.log
