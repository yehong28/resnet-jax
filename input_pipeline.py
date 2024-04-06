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

import jax
import jax.numpy as jnp
import torch
from torch.utils.data import DataLoader
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

  ds = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
  ds = map(prepare_batch_data, ds)

  return ds
