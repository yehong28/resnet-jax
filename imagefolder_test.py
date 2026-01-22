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

"""ImageFolder loader smoke test for ImageNet."""

import time
import warnings

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags

import input_pipeline
from utils import logging_util


warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('debug', False, 'Debugging mode.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


def _log_batch(name, batch):
  if isinstance(batch, dict):
    image = batch.get('image')
    label = batch.get('label')
  else:
    image, label = batch
  logging.info(
      '%s batch image shape: %s label shape: %s',
      name,
      getattr(image, 'shape', None),
      getattr(label, 'shape', None),
  )


def run_test(config):
  dataset_cfg = config.dataset
  if getattr(dataset_cfg, 'use_tfds', False):
    raise ValueError('imagefolder_test requires dataset.use_tfds=False.')
  if input_pipeline._is_gcs_path(getattr(dataset_cfg, 'root', None)):
    raise ValueError('imagefolder_test requires a local dataset.root path.')

  logging.info('dataset.root: %s', dataset_cfg.root)

  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')
  logging.info('local_batch_size: %d', local_batch_size)
  logging.info('jax.local_device_count: %d', jax.local_device_count())

  start_t = time.time()
  train_loader, steps_per_epoch = input_pipeline.create_split(
      dataset_cfg,
      local_batch_size,
      split='train',
  )
  logging.info('train create_split took %.1fs', time.time() - start_t)

  start_t = time.time()
  eval_loader, steps_per_eval = input_pipeline.create_split(
      dataset_cfg,
      local_batch_size,
      split='val',
  )
  logging.info('val create_split took %.1fs', time.time() - start_t)

  logging.info('steps_per_epoch: %d', steps_per_epoch)
  logging.info('steps_per_eval: %d', steps_per_eval)

  _log_batch('train', next(iter(train_loader)))
  _log_batch('val', next(iter(eval_loader)))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  platform.work_unit().set_task_status(
      f'process_index: {jax.process_index()}, '
      f'process_count: {jax.process_count()}'
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir'
  )

  if FLAGS.debug:
    with jax.disable_jit():
      run_test(FLAGS.config)
  else:
    run_test(FLAGS.config)


if __name__ == '__main__':
  logging_util.verbose_off()
  logging_util.set_time_logging(logging)
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
