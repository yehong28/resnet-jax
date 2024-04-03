# initialize and set up remote TPU VM

# VM_NAME=kmh-tpuvm-v3-32-1
# ZONE=europe-west4-a  # v3

VM_NAME=kmh-tpuvm-v4-32
ZONE=us-central2-b  # v4

# install packages
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

pip3 install absl-py==1.4.0
pip3 install clu==0.0.11
pip3 install flax==0.8.1
pip3 install jax[tpu]==0.4.25 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip3 install ml-collections==0.1.1
pip3 install numpy==1.26.4
pip3 install optax==0.2.1
pip3 install tensorflow==2.15.0.post1
pip3 install tensorflow-datasets==4.9.4

# sanity check
python3 -c 'import jax; print(jax.device_count())'

"

# mount NFS Filestore
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "

SHARED_FS=10.11.37.106:/kmh_nfs
MOUNT_POINT=/kmh-nfs-mount

sudo apt-get -y update
sudo apt-get -y install nfs-common

sudo mkdir -p \$MOUNT_POINT
sudo mount \$SHARED_FS \$MOUNT_POINT
sudo chmod go+rw \$MOUNT_POINT

ls \$MOUNT_POINT
"