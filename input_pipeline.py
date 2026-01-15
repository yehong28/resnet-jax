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

import math
import numpy as np
import os
import random
import time
import jax
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from absl import logging
from functools import partial


IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def _is_gcs_path(path):
  return isinstance(path, str) and (path.startswith('gs://') or path.startswith('gcs://'))


def _resolve_tfds_data_dir(dataset_cfg):
  data_dir = getattr(dataset_cfg, 'tfds_data_dir', None)
  if data_dir:
    return data_dir

  root = getattr(dataset_cfg, 'root', None)
  if _is_gcs_path(root):
    root = root.rstrip('/')
    dataset_name = getattr(dataset_cfg, 'name', None)
    suffix = f'/{dataset_name}' if dataset_name else ''
    if suffix and root.endswith(suffix):
      return root[:-len(suffix)]
    return root

  return os.environ.get('TFDS_DATA_DIR')


def _import_tfds():
  try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
  except ImportError as exc:
    raise ImportError(
      'TFDS pipeline requires tensorflow-datasets. Install it with '
      '`pip install tensorflow-datasets`.'
    ) from exc
  return tf, tfds


def _resolve_tfds_split_name(splits, split):
  if split in splits:
    return split
  if split == 'val' and 'validation' in splits:
    return 'validation'
  if split == 'validation' and 'val' in splits:
    return 'val'
  raise ValueError(f'Unknown split "{split}". Available: {list(splits)}')


class TfdsDataLoader:
  def __init__(self, ds, steps_per_epoch):
    self._ds = ds
    self._steps_per_epoch = steps_per_epoch

  def __iter__(self):
    _, tfds = _import_tfds()
    return iter(tfds.as_numpy(self._ds))

  def __len__(self):
    return self._steps_per_epoch


def prepare_batch_data(batch, batch_size=None):
  """Reformat a input batch from PyTorch Dataloader.
  
  Args:
    batch = (image, label)
      image: shape (host_batch_size, 3, height, width)
      label: shape (host_batch_size)
    batch_size = expected batch_size of this node, for eval's drop_last=False only
  """
  if isinstance(batch, dict):
    image = np.asarray(batch['image'])
    label = np.asarray(batch['label'])
  else:
    image, label = batch

  # pad the batch if smaller than batch_size
  if batch_size is not None and batch_size > image.shape[0]:
    if isinstance(batch, dict):
      image = np.concatenate([
        image,
        np.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype),
      ], axis=0)
      label = np.concatenate([
        label,
        -np.ones((batch_size - label.shape[0],), dtype=label.dtype),
      ], axis=0)
    else:
      image = torch.cat([image, torch.zeros((batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype)], axis=0)
      label = torch.cat([label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)], axis=0)

  # reshape (host_batch_size, 3, height, width) to
  # (local_devices, device_batch_size, height, width, 3)
  local_device_count = jax.local_device_count()
  if not isinstance(batch, dict):
    image = image.permute(0, 2, 3, 1)
  image = image.reshape((local_device_count, -1) + image.shape[1:])
  label = label.reshape(local_device_count, -1)

  if not isinstance(batch, dict):
    image = image.numpy()
    label = label.numpy()

  return_dict = {
    'image': image,
    'label': label,
  }

  return return_dict


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from torchvision.datasets.folder import pil_loader
def loader(path: str):
    return pil_loader(path)


def _create_tfds_split(
    dataset_cfg,
    batch_size,
    split,
):
  """Creates a split from the ImageNet dataset using TFDS."""
  tf, tfds = _import_tfds()
  if not getattr(dataset_cfg, 'name', None):
    raise ValueError('dataset.name must be set when using TFDS.')

  data_dir = _resolve_tfds_data_dir(dataset_cfg)
  if data_dir is None:
    raise ValueError(
      'TFDS data dir is not set. Set dataset.tfds_data_dir or TFDS_DATA_DIR.'
    )

  builder = tfds.builder(dataset_cfg.name, data_dir=data_dir)
  split_name = _resolve_tfds_split_name(builder.info.splits, split)
  num_examples = builder.info.splits[split_name].num_examples
  logging.info('TFDS dataset: %s split=%s data_dir=%s', dataset_cfg.name, split_name, data_dir)

  ds = builder.as_dataset(split=split_name, shuffle_files=(split == 'train'))

  def preprocess_example(example):
    image = example['image']
    label = example['label']
    image = tf.image.convert_image_dtype(image, tf.float32)
    resize_method = tf.image.ResizeMethod.BICUBIC
    if split == 'train':
      image = tf.image.resize(
        image, [IMAGE_SIZE + CROP_PADDING, IMAGE_SIZE + CROP_PADDING], method=resize_method
      )
      image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
      image = tf.image.random_flip_left_right(image)
    else:
      image = tf.image.resize(
        image, [IMAGE_SIZE + CROP_PADDING, IMAGE_SIZE + CROP_PADDING], method=resize_method
      )
      image = tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)
    image = (image - tf.constant(MEAN_RGB, dtype=image.dtype)) / tf.constant(STDDEV_RGB, dtype=image.dtype)
    return {'image': image, 'label': tf.cast(label, tf.int32)}

  ds = ds.shard(jax.process_count(), jax.process_index())
  ds = ds.map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
  if dataset_cfg.cache:
    ds = ds.cache()

  if split == 'train':
    shuffle_size = getattr(dataset_cfg, 'shuffle_buffer_size', 16 * 128)
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    steps_per_epoch = num_examples // (batch_size * jax.process_count())
  else:
    global_batch_size = batch_size * jax.process_count()
    steps_per_epoch = int(math.ceil(num_examples / global_batch_size))
    num_examples_per_host = steps_per_epoch * batch_size
    ds = ds.repeat()
    ds = ds.take(num_examples_per_host)
    ds = ds.batch(batch_size, drop_remainder=True)

  ds = ds.prefetch(tf.data.AUTOTUNE)
  return TfdsDataLoader(ds, steps_per_epoch), steps_per_epoch


def create_split(
    dataset_cfg,
    batch_size,
    split,
):
  """Creates a split from the ImageNet dataset using Torchvision Datasets.

  Args:
    dataset_cfg: Configurations for the dataset.
    batch_size: Batch size for the dataloader.
    split: 'train' or 'val'.
  Returns:
    it: A PyTorch Dataloader.
    steps_per_epoch: Number of steps to loop through the DataLoader.
  """
  use_tfds = getattr(dataset_cfg, 'use_tfds', False) or _is_gcs_path(getattr(dataset_cfg, 'root', None))
  if use_tfds:
    return _create_tfds_split(dataset_cfg, batch_size, split)

  rank = jax.process_index()
  if split == 'train':
    split_path = os.path.join(dataset_cfg.root, split)
    print(
        f'Starting ImageFolder scan for {split} at {time.time():.3f}',
        flush=True,
    )
    start_t = time.time()
    ds = datasets.ImageFolder(
      split_path,
      transform=transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
      ]),
      loader=loader,
    )
    print(
        f'Finished ImageFolder scan for {split} at {time.time():.3f} '
        f'({time.time() - start_t:.1f}s)',
        flush=True,
    )
    logging.info(ds)
    sampler = DistributedSampler(
      ds,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=True,
    )
    it = DataLoader(
      ds, batch_size=batch_size, drop_last=True,
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None,
      pin_memory=dataset_cfg.pin_memory,
      persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
  elif split == 'val':
    split_path = os.path.join(dataset_cfg.root, split)
    print(
        f'Starting ImageFolder scan for {split} at {time.time():.3f}',
        flush=True,
    )
    start_t = time.time()
    ds = datasets.ImageFolder(
      split_path,
      transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE + CROP_PADDING, interpolation=3),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
      ]),
      loader=loader,
    )
    print(
        f'Finished ImageFolder scan for {split} at {time.time():.3f} '
        f'({time.time() - start_t:.1f}s)',
        flush=True,
    )
    logging.info(ds)
    '''
    The val has 50000 images. We want to eval exactly 50000 images. When the
    batch is too big (>16), this number is not divisible by the batch size. We
    set drop_last=False and we will have a tailing batch smaller than the batch
    size, which requires modifying some eval code.
    '''
    sampler = DistributedSampler(
      ds,
      num_replicas=jax.process_count(),
      rank=rank,
      shuffle=False,  # don't shuffle for val
    )
    it = DataLoader(
      ds, batch_size=batch_size,
      drop_last=False,  # don't drop for val
      worker_init_fn=partial(worker_init_fn, rank=rank),
      sampler=sampler,
      num_workers=dataset_cfg.num_workers,
      prefetch_factor=dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None,
      pin_memory=dataset_cfg.pin_memory,
      persistent_workers=True if dataset_cfg.num_workers > 0 else False,
    )
    steps_per_epoch = len(it)
  else:
    raise NotImplementedError

  return it, steps_per_epoch
