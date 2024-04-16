import jax


def is_distributed():
    return (jax.process_count() > 1)


def get_world_size():
    return jax.process_count()


def get_rank():
    return jax.process_index()


def is_main_process():
    return (jax.process_index() == 0)