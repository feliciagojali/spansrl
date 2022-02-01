import tensorflow as tf
from tensorflow.keras.layers import Layer

class Pruning(Layer):
    def __init__(self, max, **kwargs):
        self.max = max
        super(Pruning, self).__init__(**kwargs)
      
    def call(self, input): # Shape input: (batch_size, max_tokens, 1)
        scores = input
        sorted_scores = tf.argsort(scores, direction='DESCENDING', axis=1)
        sorted_index = tf.squeeze(sorted_scores, axis=-1)    
        filtered_idx = tf.gather(sorted_index, tf.range(self.max), axis=1)
        # Shape: (batch_size, max_candidates)
        return filtered_idx

