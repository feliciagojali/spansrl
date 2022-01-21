import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Softmax

class Attention(Layer):
    def __init__(self, config, span_idx, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.num_heads = 1
        self.dense = Dense(self.num_heads, activation='relu', name='head_scores')
        self.softmax = Softmax(axis=2)
        self.max_arg_span = config['max_arg_span']
        self.max_tokens = config['max_tokens']
        self.start, self.end, _ = span_idx
                  
    def call(self, input): # Shape input: (batch_size, max_tokens, emb)
        # Shape span_indices: (num_spans, max_arg_span)
        span_indices = tf.minimum(
          tf.expand_dims(tf.range(self.max_arg_span), 0) + tf.expand_dims(self.start, 1),
          self.max_tokens - 1)
        # Shape span_emb: (batch_size, num_spans, max_arg_span, emb)
        span_emb = tf.gather(input, span_indices, axis=1)

        # Shape head_scores: (batch_size, max_tokens, 1)
        head_scores = self.dense(input)
        # Shape span_width:: (num_spans)
        span_width = tf.add(tf.subtract(self.end, self.start), 1)
        # Shape span_indices_mask = (num_spans, max_arg_span)
        span_indices_mask = tf.sequence_mask(span_width, self.max_arg_span, dtype=tf.float32)
        span_indices_log_mask = tf.math.log(span_indices_mask)

        # Shape span_head: (batch_size, num_spans, max_arg_span, 1)
        span_head = tf.gather(head_scores, span_indices, axis=1) + tf.tile(tf.expand_dims(span_indices_log_mask, -1), [1, 1, self.num_heads])
        span_head = self.softmax(span_head)
        
        # Shape out: (batch_size, num_spans, emb)
        out = tf.reduce_sum(span_emb * span_head, 2)
        return out