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

"""TFDS loader smoke test for ImageNet."""

import time
start = time.time()
print("start imports", flush=True)
from absl import app
from absl import flags
print("absl", flush=True)
import jax
print("ajx", flush=True)
from ml_collections import config_flags
print(f"done={time.time()-start}", flush=True)

import input_pipeline
print(f"ez",flush=True)

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


def _log_batch(name, batch):
  image = batch.get('image')
  label = batch.get('label')
  print(
      f'{name} batch image shape: {getattr(image, "shape", None)} '
      f'label shape: {getattr(label, "shape", None)}',
      flush=True,
  )


def run_test(config):
  dataset_cfg = config.dataset
  use_tfds = getattr(dataset_cfg, 'use_tfds', False) or input_pipeline._is_gcs_path(
      getattr(dataset_cfg, 'root', None)
  )
  if not use_tfds:
    raise ValueError('tfds_test requires dataset.use_tfds=True or a gcs:// root.')

  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  print(f'dataset.name: {getattr(dataset_cfg, "name", None)}', flush=True)
  print(f'dataset.root: {getattr(dataset_cfg, "root", None)}', flush=True)
  print(f'dataset.tfds_data_dir: {getattr(dataset_cfg, "tfds_data_dir", None)}', flush=True)
  print(f'local_batch_size: {local_batch_size}', flush=True)
  print(f'jax.local_device_count: {jax.local_device_count()}', flush=True)

  start_t = time.time()
  train_loader, steps_per_epoch = input_pipeline.create_split(
      dataset_cfg,
      local_batch_size,
      split='train',
  )
  print(f'train create_split took {time.time() - start_t:.1f}s', flush=True)

  start_t = time.time()
  eval_loader, steps_per_eval = input_pipeline.create_split(
      dataset_cfg,
      local_batch_size,
      split='val',
  )
  print(f'val create_split took {time.time() - start_t:.1f}s', flush=True)

  print(f'steps_per_epoch: {steps_per_epoch}', flush=True)
  print(f'steps_per_eval: {steps_per_eval}', flush=True)

  _log_batch('train', next(iter(train_loader)))
  _log_batch('val', next(iter(eval_loader)))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_test(FLAGS.config)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config'])
  app.run(main)
