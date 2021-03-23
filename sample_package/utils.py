import logging
import os
import random
import sys
from typing import Callable, Iterable, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Schedule learning rate linearly from max_learning_rate to min_learning_rate. """

    def __init__(
        self,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: Optional[float] = None,
        warmup_steps: Optional[int] = None,
    ):
        self.warmup_steps = int(total_steps * warmup_rate) + 1 if warmup_steps is None else warmup_steps
        self.increasing_delta = max_learning_rate / self.warmup_steps
        self.decreasing_delta = (max_learning_rate - min_learning_rate) / (total_steps - self.warmup_steps)
        self.max_learning_rate = tf.cast(max_learning_rate, tf.float32)
        self.min_learning_rate = tf.cast(min_learning_rate, tf.float32)
        self.total_steps = total_steps

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        lr = tf.reduce_min([step * self.increasing_delta, self.max_learning_rate - step * self.decreasing_delta])
        return tf.reduce_max([lr, self.min_learning_rate])


def get_logger() -> logging.Logger:
    """ Return logger for logging """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def path_join(*paths: Iterable[str]) -> str:
    """ Join paths to string local paths and google storage paths also """
    if paths[0].startswith("gs://"):
        return "/".join((path.strip("/") for path in paths))
    return os.path.join(*paths)


def set_random_seed(seed: int):
    """ Set random seed for random / numpy / tensorflow """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_device_strategy(device) -> tf.distribute.Strategy:
    """ Return tensorflow device strategy """
    # Use TPU
    if device.upper() == "TPU":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=os.environ["TPU_NAME"])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        return strategy

    # Use GPU
    if device.upper() == "GPU":
        devices = tf.config.list_physical_devices("GPU")
        if len(devices) == 0:
            raise RuntimeError("Cannot find GPU!")
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
        if len(devices) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")

        return strategy

    # Use CPU
    return tf.distribute.OneDeviceStrategy("/cpu:0")
