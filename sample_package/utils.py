import tensorflow as tf
from tensorflow.keras import backend as K


def learning_rate_scheduler(num_epochs, max_learning_rate, min_learninga_rate=1e-7):
    lr_delta = (max_learning_rate - min_learninga_rate) / num_epochs

    def _scheduler(epoch, lr):
        return lr - lr_delta

    return _scheduler
