import tensorflow as tf
from tensorflow.keras.layers import Layer
tf.random.set_seed(42)

class SpanEndpointsLength(Layer): 
    # To get endpoints representation for each span to build span representations
    def __init__(self, span_idx, **kwargs):
        super(SpanEndpointsLength, self).__init__(**kwargs)
        self.start, self.end, self.width = span_idx

    def call(self, input): #inputs: [batch_size, max_tokens, emb] # embeddings
        # Shape span_start, span_end: [batch_size, num_spans, emb] # emb of start and end id of every span
        span_start = tf.gather(input, self.start, axis=1, name='span_start')
        span_end = tf.gather(input, self.end, axis=1, name='span_end')

        batch_size = tf.shape(input)[0]
        # Shape width: (num_spans) 
        width = tf.convert_to_tensor(self.width)
        expanded_width = tf.expand_dims(width, 0) # Shape: (1, num_spans)
        span_length = tf.tile(expanded_width, [batch_size, 1]) # Shape: (batch_size, num_spans)

        out = [span_start, span_end, span_length]
        return out

class PredicateArgumentEmb(Layer): 
    # To compute embedding of each predicate-argument pair
    def __init__(self, **kwargs):
        super(PredicateArgumentEmb, self).__init__(**kwargs)

    def call(self, inputs): 
        # Shape arg_emb: (batch_size, num_args, emb)
        # Shape pred_emg: (batch_size, num_preds, emb)
        arg_emb, pred_emb = inputs

        arg_emb_expanded = tf.expand_dims(arg_emb, 1) # Shape: (batch_size, 1, num_args, emb)
        pred_emb_expanded = tf.expand_dims(pred_emb, 2) # Shape: (batch_size, num_preds, 1, emb)
        
        num_spans = arg_emb_expanded.shape[2]
        num_preds = pred_emb_expanded.shape[1]
        arg_emb_tiled = tf.tile(arg_emb_expanded, [1, num_preds, 1, 1])  # Shape: (batch_size, num_preds, num_args, emb)
        pred_emb_tiled = tf.tile(pred_emb_expanded, [1, 1, num_spans, 1])  # Shape: (batch_size, num_preds, num_args, emb)

        pair_emb_list = [pred_emb_tiled, arg_emb_tiled]
        pair_emb = tf.concat(pair_emb_list, 3)  # Shape: (batch_size, num_preds, num_args, pred_emb+arg_emb)

        out = pair_emb
        return out

class NullScores(Layer):
    # To generate scores for null labels which is a constant 0
    def __init__(self, **kwargs):
        super(NullScores, self).__init__(**kwargs)

    def call(self, inputs): 
        # Shape arg_emb: (batch_size, num_args, _)
        # Shape pred_emg: (batch_size, num_preds, _)
        arg_emb, pred_emb = inputs

        num_args = arg_emb.shape[1]
        num_preds = pred_emb.shape[1]

        batch_size = tf.shape(arg_emb)[0]
        out = tf.zeros([batch_size, num_preds, num_args, 1])

        return out