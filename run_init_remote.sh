# initialize and set up remote TPU VM

VM_NAME=kmh-tpuvm-v3-32-1
ZONE=europe-west4-a

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
cd ~/resnet_jax
pip install -r requirements.txt
"