from typing import Optional

import tensorflow as tf


# Remove SampleModel and replace
class SampleModel(tf.keras.Model):
    def __init__(self, embedding_dim: [int]):
        super(SampleModel, self).__init__()

        self.embedding = tf.keras.layers.Embedding(30, embedding_dim)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None):
        output = self.embedding(inputs)
        return output
