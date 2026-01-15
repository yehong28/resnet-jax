# Jax training with PyTorch dataloaders

Written by Congyue Deng, Kaiming He.

### Notes
PyTorch DataLoader

- This branch is built on top of the main branch: https://github.com/KaimingHe/resnet_jax/tree/main, which is based on the TFDS dataloader.

- For debugging, use the flag `--debug=True` to call `with jax.disable_jit():` which disables jax compilation. Be careful, this may increase memory consumption

- Increase `num_workers` and `prefetch_factor`. A thorough discussion: https://github.com/pytorch/xla/issues/2690

## Step-by-Step Instruction

### Introduction

Before you start, please read [He Vision Group's GCP/TPU Wiki](https://github.com/hevision/wiki_tpu).

JAX and TPU are useful and great. But GCP (Google Cloud Platform) takes some time to learn. This tutorial will walk you through some basic concepts about JAX, TPU, and GCP.

### SSH into your TPU VM

SSH into your TPU VM that has 8 TPUs (say, v4-8). Every 8 TPUs are in one node. Unlike PyTorch, in JAX, we only need one Python process per node. This **single-node** TPU VM works as your dev machine.

If you have followed [He Vision Group's GCP/TPU Wiki](https://github.com/hevision/wiki_tpu), you should be able to SSH into a TPU VM by running the following command in your **laptop**:
```shell
DEV_VM=kmh-tpuvm-v4-8-1
ssh $DEV_VM
```
Here `kmh-tpuvm-v4-8-1` is the TPU VM's name. In the following, we assume you are already in your dev TPU VM.

**Please check this "[manual SLRUM](https://docs.google.com/spreadsheets/d/1vDpP7eTkYRwWYs2fo-9dJwxHKvN6SDp5iC1cM1H3FDU/edit?usp=sharing)" spreadsheet for available TPU VMs. Do NOT use other people's TPU VM, as it may cause conflict in TPU using** (see [issues](#tpu-is-in-use)).

### Mount NFS Filestore

All our TPU VMs will access to a shared file system for data and code. The TPU VMs are not permanent; the shared file system is. We use NFS Filestore from GCP. In your TPU VM, run the following to install the tool for mounting:
```shell
sudo apt-get -y update
sudo apt-get -y install nfs-common
```
Then you can mount a disk by:
```shell
sudo mkdir -p /kmh-nfs-us-mount
sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount
ls /kmh-nfs-us-mount
```
Here `/kmh-nfs-us-mount` is like a local dir that can be accessed from your TPU VM.

**Note**: The actual name and address may change. Check the current ones in [this page](https://console.cloud.google.com/filestore/instances?referrer=search&project=he-vision-group)

### Mount NFS Filestore (*** with Pytorch Loader ***)

To use Pytorch's dataloader, we need high-throughput SSD disk located in the same region as your TPU VMs. Mount the following SSD disk in `us-central2-b` when you are using **TPU v4** machines for your remote jobs:
```shell
sudo mkdir -p /kmh-nfs-ssd-us-mount
sudo mount -o vers=3 10.97.81.98:/kmh_nfs_ssd_us /kmh-nfs-ssd-us-mount
sudo chmod go+rw /kmh-nfs-ssd-us-mount
ls /kmh-nfs-ssd-us-mount
```

The ImageNet dataset, in their per-image raw formats for Pytorch dataloader, is in `/kmh-nfs-ssd-us-mount/data/imagenet`.

### Manage your code

We recommend you to put your code in the NFS mount, not in the local TPU VM. Then your code can be run in different machines. Create a dir in the mount and clone this repo:
```
sudo chmod go+rw /kmh-nfs-us-mount/code
mkdir /kmh-nfs-us-mount/code/$USER/
cd /kmh-nfs-us-mount/code/$USER/
git clone https://github.com/KaimingHe/resnet_jax.git
cd resnet_jax
```


### Install packages

Run `pip install -r requirements.txt` to install the minimal requirements.

- torch, torchvision
- tensorflow-datasets (required for TFDS/GCS datasets)
- [2024.4] ml_collections not working with Python>=3.12 (imp deprecated): https://github.com/google/ml_collections/pull/28. To fix this problem, install ml_collections from:
  ```
  pip install git+https://github.com/danielkelshaw/ml_collections.git
  ```

### Run single-node training (*** with Pytorch Loader ***)

In your TPU VM, run the following for sanity check of JAX/TPU:
```shell
python3 -c 'import jax; print(jax.device_count())'
```
It should output `4` for TPU v4-8 (or `8` for TPU v3-8). If you are waiting forever, it implies you are not in a single-node TPU VM; if you see some error with `/dev/accel0`, it implies the TPUs are run by someone else.

Now, run a single-node training:
```shell
export TFDS_DATA_DIR='gs://kmh-gcp-us-central2/tensorflow_datasets'
PWD=$(pwd)
python3 main.py \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.use_tfds=True \
    --config.dataset.name=imagenet_fake \
    --config.dataset.root='gs://kmh-gcp-us-central2/tensorflow_datasets/imagenet_fake' \
    --config.batch_size=1024 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=10 \
    --config.model='_ResNet1'
```

This command can also be found in `run_script.sh`, which is what I use to run local dev jobs.

**Note:**
- `TFDS_DATA_DIR` points to the base TFDS bucket; `dataset.root` may point to the dataset subdir and will be normalized.
- `imagenet_fake` is a small TFDS dataset for fast debugging.
- `_ResNet1` is a tiny ResNet for fast debugging.

The first few iterations of the log look like this:
<img width="1155" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/a181d0f2-591f-47f9-9c87-f884feb0ee92">


You can see that the speed is not ideal, even though we train a tiny `_ResNet1`. **Data loading time with Pytorch loader is a major bottleneck for small models** (including even ResNet-50, 101, 152; see below).


### Run multi-node training

#### Concept

Any TPU VM with more than 8 TPUs is conceptually a multi-node machine. For example, `v4-32` is conceptually 4 nodes, and we will do the same thing for each node. To have some sense of it, run the following:
```
VM_NAME=kmh-tpuvm-v4-32-1
ZONE=us-central2-b

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "echo HelloWorld"
```
And you will see:
```
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
HelloWorld
HelloWorld
HelloWorld
HelloWorld
```

**Note:**
- When you run this demo, make sure `kmh-tpuvm-v4-32-1` is available.


#### Install packages in remote nodes

We need to install all packages and mount NFS in our remote TPU VM:

Open `run_init_remote.sh`, change `VM_NAME` and `ZONE` into your remote TPU VM, say: `VM_NAME=kmh-tpuvm-v4-32-1` and ` ZONE=us-central2-b`.

Then run `source run_init_remote.sh`. This will install packages and mount NFS in the remote TPU VM.


#### Manage your jobs

The "remote" TPU VM is like your dev TPU VM. Conceptually, we need to run the same code for all nodes in the multi-node TPU VM, one Python process per node. So first of all, we need nodes to access to the same copy of code, and the same destination of artifacts (logs and checkpoints).

In your **dev** TPU VM (say, `v4-8`), run:
```
mkdir /kmh-nfs-us-mount/logs/$USER/
mkdir /kmh-nfs-us-mount/staging/$USER/
sudo chmod 777 /kmh-nfs-us-mount/logs/$USER
sudo chmod 777 /kmh-nfs-us-mount/staging/$USER
```
Here, `logs` is the dir to the remote job's artifacts, and `staging` is the dir for staged (cached) **copies** of codes that wil be run in remote TPU VM.

<!-- **Note:**
- You may notice that here the artifacts are in `/kmh-nfs-ssd-us-mount` (zone=`us`). In case you have big artifacts (e.g., very large checkpoints), you may want to use the mount in the same zone as your remote TPU VM (`/kmh-nfs-ssd-eu-mount` for TPU v3 in zone=`eu`):
```
mkdir /kmh-nfs-ssd-eu-mount/logs/$USER/
sudo chmod 777 /kmh-nfs-ssd-eu-mount/logs/$USER
``` -->


#### Run a remote job
  
You may open `run_remote.sh` to see how a remote job is run. Conceptually, the essential part is (you don't need to run this line):
```
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "python3 main.py"
```
Here, `--worker=all` means the same command `python3 main.py` will be run in all nodes.

The file `run_remote.sh` alone does not take effect; instead, we use `run_staging.sh` to kick off a remote job. You may open `run_staging.sh` and see the process. Basicall, it will copy the current repo dir (in you dev TPU VM) to a hashed dir in `/kmh-nfs-us-mount/staging/$USER/` and `cd` into it, then it will run the `run_remote.sh` file in the staging dir.

In sum, you only need to run `run_staging.sh` in your **dev** TPU VM by:
```
source run_staging.sh
```
Then you can kick off your remote job.

The following is the beginning of the output you may see:
```
Staging files...
Done staging.
Current dir: /kmh-nfs-ssd-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
kmh-tpuvm-v3-32-2 europe-west4-a
Log dir: /kmh-nfs-ssd-eu-mount/logs/kaiminghe/resnet/20240419_000859_d4awst_kmh-tpuvm-v3-32-2_tpu_b1024_lr0.1_ep100_torchvision_ep1000x
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
Current dir: /kmh-nfs-ssd-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
Current dir: /kmh-nfs-ssd-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
Current dir: /kmh-nfs-ssd-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
Current dir: /kmh-nfs-ssd-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
```

#### Cancel a remote job (NOTE: NOT DELETING TPU VM)

Oftentimes you may not want your remote job to finish training (e.g., you found a bug in your code). You may Ctrl+C with your remote job, but it only shuts down the ssh client running it, not the the job itself. If you will run anothe job in the same remote TPU VM, you may see some error related to `/dev/accel0`, which implies TPUs have been running by some other jobs.

When this happens, you may use `run_kill_remote.sh` to kill the jobs in remote TPU VM. Run `source run_kill_remote.sh` in your **dev** TPU VM.

**Caution 1**: Before you kill the job, make sure the remote TPU VM is the one you want to handle. Running this `run_kill_remote.sh` on another TPU VM can kill other people's jobs.

**Caution 2**: This command in `run_kill_remote.sh` is only about killing a job. It won't delete the TPU VM. And we suggest NOT to delete TPU VM unless necessary.


### Tensorboard Monitoring

To access the Tensorboard profile generated by the job, run the following in your **laptop** when SSH into your dev TPU VM:
```shell
ssh $DEV_VM -L 6060:localhost:6060
```
You may change `6060` into whatever number you like.

Assume Tensorboard has been installed in the dev TPU VM. If the alias of `tensorboard` is not defined, you may run `alias tensorboard='python3 -m tensorboard.main'` to define it (you may want to put this into your `.zshrc` or `.bashrc`).

In your TPU VM, run the following:
```
tensorboard --port=6060 --logdir_spec=\
v3-32-2_tpu_b1024_lr0.1_ep100_torchvision:/kmh-nfs-ssd-eu-mount/logs/kaiminghe/resnet/20240419_020919_nevsyj_kmh-tpuvm-v3-32-2_tpu_b1024_lr0.1_ep100_torchvision
```

Then open `http://localhost:6060/#scalars` in your laptop. You can see the tensorboard profile:

<img width="587" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/39829a1f-6258-4062-aed3-71fdc88891f5">

(Notes: now, since the path `/kmh-nfs-ssd-eu-mount` no longer exists, the website will show "No dashboards are active for the current data set". Same as the run below.)

**Note:**
- Here, the metrics starting with `ep_` will have `epochs` (x1000) as the x-axis. For example, x-axis with 100k just means 100 epochs. This is useful for calibrating different runs with different batch sizes. 


Here is another run with 128 TPUs (`v3-128`, 16 nodes):
```
tensorboard --port=6060 --logdir_spec=\
v3-128-1_tpu_b8192_lr0.1_ep100_torchvision:/kmh-nfs-ssd-eu-mount/logs/kaiminghe/resnet/20240419_023004_itp67c_kmh-tpuvm-v3-128-1_tpu_b8192_lr0.1_ep100_torchvision
```

It is about 3x faster than `v3-32`. Note that data loading is still the bottleneck.
<img width="610" alt="Screenshot 2024-04-19 at 10 17 22 AM" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/3592e263-d509-4d62-a087-28c6c3c8621f">


### *** When to use Pytorch Loader? ***

- As you may see, when using the Pytorch Loader, training ResNet-50 is 2-3x slower than using TFDS. This is because data loader time is the bottleneck.

- Because of this, training ResNet-50, -101, -152, -200 basically has the same time. See below, with 128 TPUs (`v3-128`) and a batch size of 4096. The training time hasn't overlapped the loading time until ResNet-200.

<img width="621" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/12e96992-53d3-4fa6-b8fe-41f9043b9f12">


- Based on this observation, Pytorch dataloader is most favored when training **large** models (like ViT), and when using fancy data augmentation (like [timm](https://pypi.org/project/timm/)) which is readily available in Pytorch.

### *** Numerical Reproducibility ***

In JAX, the random seed is treated as part of the input to the function; as such, the JAX computation is fully deterministic. In TPUs, the computation is fully deterministic (while in GPU it is not). Using Pytorch Loader, the multi-process dataloader (in both single-/multi-node settings) has process/worker-specific random seed control, and can be made fully deterministic.

When using the same random seed, the same versions of packages, and the same hyper-parameters, running the same job would give **100% numerically exact same result**, up to every single digit and every iteration.

Below are the training loss curves of two jobs. They are exactly the same. Even the running time is very similar.

<img width="1159" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/b9a58d7d-69cd-45b8-b419-79eda6a0b062">

## Issues

### TPU Is In Use

Unlike PyTorch+GPU, in JAX+TPU, on one TPU VM there can be **at most one program among all users** that is using the TPU. You may encounter error message like these:

- error message type 1:
  ```
  Traceback (most recent call last):
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 886, in backends
      backend = _init_backend(platform)
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 977, in _init_backend
      backend = registration.factory()
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 164, in tpu_client_timer_callback
      client = xla_client.make_tpu_client(_get_tpu_library_path())
    File "/usr/lib/python3.10/site-packages/jaxlib/xla_client.py", line 207, in make_tpu_client
      return make_tfrt_tpu_c_api_client()
    File "/usr/lib/python3.10/site-packages/jaxlib/xla_client.py", line 128, in make_tfrt_tpu_c_api_client
      initialize_pjrt_plugin('tpu')
    File "/usr/lib/python3.10/site-packages/jaxlib/xla_client.py", line 176, in initialize_pjrt_plugin
      _xla.initialize_pjrt_plugin(plugin_name)
  jaxlib.xla_extension.XlaRuntimeError: ABORTED: The TPU is already in use by another process probably owned by another user. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. If you still get this message, run "$ sudo rm /tmp/libtpu_lockfile".

  During handling of the above exception, another exception occurred:

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 1089, in devices
      return get_backend(backend).devices()
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 1023, in get_backend
      return _get_backend_uncached(platform)
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 1002, in _get_backend_uncached
      bs = backends()
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 902, in backends
      raise RuntimeError(err_msg)
  RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by another process probably owned by another user. Run "$ sudo lsof -w /dev/accel0" to figure out which process is using the TPU. If you still get this message, run "$ sudo rm /tmp/libtpu_lockfile". (set JAX_PLATFORMS='' to automatically choose an available backend)
  ```
  which implies that **another person** is using the TPU VM;

- error message type 2:
  ```
  Traceback (most recent call last):
  File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 886, in backends
    backend = _init_backend(platform)
  File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 977, in _init_backend
    backend = registration.factory()
  File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 164, in tpu_client_timer_callback
    client = xla_client.make_tpu_client(_get_tpu_library_path())
  File "/usr/lib/python3.10/site-packages/jaxlib/xla_client.py", line 207, in make_tpu_client
    return make_tfrt_tpu_c_api_client()
  File "/usr/lib/python3.10/site-packages/jaxlib/xla_client.py", line 128, in make_tfrt_tpu_c_api_client
    initialize_pjrt_plugin('tpu')
  File "/usr/lib/python3.10/site-packages/jaxlib/xla_client.py", line 176, in initialize_pjrt_plugin
    _xla.initialize_pjrt_plugin(plugin_name)
  jaxlib.xla_extension.XlaRuntimeError: ABORTED: The TPU is already in use by process with pid 2239507. Not attempting to load libtpu.so in this process.

  During handling of the above exception, another exception occurred:

  Traceback (most recent call last):
    File "<string>", line 1, in <module>
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 1089, in devices
      return get_backend(backend).devices()
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 1023, in get_backend
      return _get_backend_uncached(platform)
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 1002, in _get_backend_uncached
      bs = backends()
    File "/usr/lib/python3.10/site-packages/jax/_src/xla_bridge.py", line 902, in backends
      raise RuntimeError(err_msg)
  RuntimeError: Unable to initialize backend 'tpu': ABORTED: The TPU is already in use by process with pid 2239507. Not attempting to load libtpu.so in this process. (set JAX_PLATFORMS='' to automatically choose an available backend)
  ```
  which implies that **your another process** is using the TPU VM.

- error message type 3:
  ```
  Could not open the log file '/tmp/tpu_logs/tpu_driver.t1v-n-d0a278b1-w-0.zhh.log.INFO.20250927-192806.2244481': Permission denied
  Could not open any log file.
  (... Usually repeated many times ...)
  ```
  which implies that **although TPU is not in use, someone else has used it in the past, leading you to lose the permission of the TPU log files**. In such case, your code will not fail; instead it will run gracefully, only the output will be polluted by these messages.

#### Handling the Problems

- ⚠️⚠️⚠️ Before doing **anything**, please **check the manual slurm page [here](https://docs.google.com/spreadsheets/d/1vDpP7eTkYRwWYs2fo-9dJwxHKvN6SDp5iC1cM1H3FDU/edit?usp=sharing) to make sure you have ssh'ed into your own VM**. ⚠️⚠️⚠️
- ⚠️⚠️⚠️ In **case 1 and 2**, **DO NOT** directly run the suggested command `sudo rm /tmp/libtpu_lockfile`, or kill the PID provided (in error message type 2). As the TPU is shared among users, doing this **may kill the program that other people is running**.

  Instead, first use `sudo lsof -w /dev/accel0` to find out who and what process is using the TPU (the user is displayed in the output), and contact the relevant people.
- In **case 1 and 2**, once you are sure that you can kill the program (i.e. get the permission of the owner, or it is program run by yourself), you can do **either** of the two:
  - kill the program (**preferred**): `sudo kill -9 <PID>`
  - run:
    ```shell
    sudo rm /tmp/libtpu_lockfile
    sudo rm -rf /tmp/tpu_logs
    ```
  
  The difference: the first method kills the program elegantly; the second make the program lose the access to TPU, hence it may fail, but anyway it can't be using the TPU anymore (so you can use it).
- In **case 3**, just run
  ```shell
  sudo rm -rf /tmp/tpu_logs
  ```
  Alternatively, you may use
  ```shell
  your_command 2>&1 | grep -v Could
  ```
  which will not cause any real changes(as mentioned, in case 3 there is no problem with the program; it is just the output is polluted).
- It may also be the case that you can't use the TPU (others is use it). However, you can still **debug** your code using CPU:
  ```shell
  JAX_PLATFORMS=cpu python your_program.py
  ```
  Using cpu to debug the code can be much efficient.
