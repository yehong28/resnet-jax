# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet input pipeline."""

import numpy as np
import random
import jax
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def prepare_batch_data(batch):
  """Reformat a input batch from PyTorch Dataloader.
  
  Args:
    batch = (image, label)
      image: shape (host_batch_size, 3, height, width)
      label: shape (host_batch_size)
  """
  image, label = batch

  # reshape (host_batch_size, 3, height, width) to
  # (local_devices, device_batch_size, height, width, 3)
  local_device_count = jax.local_device_count()
  image = image.permute(0, 2, 3, 1)
  image = image.reshape((local_device_count, -1) + image.shape[1:])
  label = label.reshape(local_device_count, -1)

  image = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(image.contiguous()))
  label = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(label.contiguous()))

  return_dict = {
    'image': image,
    'label': label,
  }

  return return_dict


# def collate_fn(batch):
#   batch = default_collate(batch)
#   batch = prepare_batch_data(batch)
#   return batch


def worker_init_fn(worker_id):
    seed = worker_id
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def create_split(
    dataset_cfg,
    batch_size,
    split,
    input_dtype=torch.float32,
):
  """Creates a split from the ImageNet dataset using Torchvision Datasets.

  Args:
    TODO: Add args explanation.
  Returns:
    TODO: Add returns explanation.
  """
  ds = datasets.ImageNet(
    dataset_cfg.root,
    split=split,
    transform=transforms.Compose([
      transforms.RandomResizedCrop(IMAGE_SIZE),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
      transforms.ConvertImageDtype(input_dtype),
  ]))

  if split == 'train':
    sampler = DistributedSampler(
      ds,
      num_replicas=get_world_size(),
      rank=get_rank(),
      shuffle=True,
    )
    it = DataLoader(
      ds, batch_size=batch_size, drop_last=True,
      #collate_fn=collate_fn,
      worker_init_fn=worker_init_fn,
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor,
      pin_memory=dataset_cfg.pin_memory,
    )
  else:
    it = DataLoader(
      ds, batch_size=batch_size, shuffle=True, drop_last=True,
      #collate_fn=collate_fn,
      worker_init_fn=worker_init_fn,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor,
      pin_memory=dataset_cfg.pin_memory,
    )

  steps_per_epoch = len(it)
  it = map(prepare_batch_data, it)

  return it, steps_per_epoch
