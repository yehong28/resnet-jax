## Jax training with PyTorch dataloaders

Work in progress. Written by Congyue Deng, Kaiming He.

### Notes
PyTorch DataLoader
- `flax.jax_utils.prefetch_to_device()` not necessary for TPU and CPU
- For debugging, use the flag `--debug=True` to call `with jax.disable_jit():` which disables jax compilation. Be careful, this may increase memory consumption
- bfloat16 datatype not supported for numpy (without the Tensorflow extension `RegisterNumpyBfloat16`?). Directly convert torch tensors to jnp arrays: https://github.com/samuela/torch2jax/blob/bd7bd9c95253c89ffb7a25cc0ff2ccb296f6cfbf/torch2jax/__init__.py#L12
- Increase `num_workers` and `prefetch_factor`. A thorough discussion: https://github.com/pytorch/xla/issues/2690
- Random seed control
  - DataLoader: `worker_init_fn`
  - Epochs: `torch.utils.data.distributed.DistributedSampler` for distirbuted training.
    <s>The training curve is exactly the same with or without `train_loader.sampler.set_epoch(epoch)`, and need to rewrite `map(dataloader)` to `collate_fn` which seems decrease the data loading speed, so it is removed.</s>
    In distributed (multi-node) loading, every single "node" (to be precisely, "process", but in JAX, one node has one process) maintains only a subset of the data: with N nodes, each node has 1281167 / N samples cached. if there's no `set_epoch`, the subsets across different nodes won't be shuffled.

### Packages
- torch, torchvision
- [2024.4] ml_collections not working with Python>=3.12 (imp deprecated): https://github.com/google/ml_collections/pull/28. To fix this problem, install ml_collections from:
  ```
  pip install git+https://github.com/danielkelshaw/ml_collections.git
  ```


