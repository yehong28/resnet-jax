# initialize and set up remote TPU VM

VM_NAME=kmh-tpuvm-v3-32-2
ZONE=europe-west4-a  # v3

# install packages
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

sudo apt-get -y update
sudo apt-get -y install nfs-common

pip3 install absl-py==1.4.0
pip3 install clu==0.0.11
pip3 install flax==0.8.1
pip3 install jax[tpu]==0.4.25 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install ml-collections==0.1.1
pip3 install numpy==1.26.4
pip3 install optax==0.2.1
pip3 install torch==2.2.2
pip3 install torchvision==0.17.2

"

# sanity check
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
python3 -c 'import jax; print(jax.device_count())'
"

# # mount NFS Filestore
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

sudo apt-get -y update
sudo apt-get -y install nfs-common

sudo mkdir -p /kmh-nfs-mount
sudo mount -o vers=3 10.11.37.106:/kmh_nfs /kmh-nfs-mount
sudo chmod go+rw /kmh-nfs-mount
ls /kmh-nfs-mount

sudo mkdir -p /kmh-nfs-us-mount
sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount
ls /kmh-nfs-us-mount
"