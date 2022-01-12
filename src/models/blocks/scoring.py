from typing import final
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, Add, Concatenate

from models.blocks.functions import NullScores

class Scoring(Layer):
    def __init__(self, config, output_size, **kwargs):
        super(Scoring, self).__init__(**kwargs)
        self.dense = []
        self.dropout = []
        for _ in range(config['unary_depth']):
            self.dense.append(Dense(config['unary_units'], activation='relu'))
            self.dropout.append(Dropout(config['dropout_value']))
        
        self.dense_n = Dense(output_size)

    def call(self, input): # Shape input: (batch_size, max_tokens (span/pred), emb)
        current_input = input
        for dense, dropout in zip(self.dense, self.dropout):
            # Shape d: (batch_size, max_tokens, dense_units)
            d = dense(current_input)
            d = dropout(d)

            current_input = d
            out = current_input

        # Shape out: (batch_size, max_tokens, 1)
        out = self.dense_n(out)
        return out

class ComputeScoring(Layer):
    def __init__(self, num_labels, **kwargs):
        super(ComputeScoring, self).__init__(**kwargs)
        self.num_labels = num_labels
        self.add = Add(name='total_score')
        self.null_score = NullScores(name="null_scores")
        self.concatenate = Concatenate(axis=-1)

    def call(self, inputs):
        # Shape arg_unary_score, pred_unary_score: (batch_size, num_candidates, 1)
        # Shape pair_score: (batch_size, num_args, num_preds, num_labels)
        # Shape biaffine_score: (batch_size, num_args, num_preds, num_labels)
        arg_unary_score, pred_unary_score, pair_score, biaffine_score = inputs

        num_args = arg_unary_score.shape[1]
        num_preds = pred_unary_score.shape[1]
        # Shape arg_score, pred_score: (batch_size, num_args, num_preds, num_labels)
        arg_score = tf.tile(tf.expand_dims(arg_unary_score, 2), [1, 1, num_preds, self.num_labels])
        pred_score = tf.tile(tf.expand_dims(pred_unary_score, 1), [1, num_args, 1, self.num_labels])

        # Shape total_score: (batch_size, num_args, num_preds, num_labels)
        total_score = self.add([arg_score, pred_score, pair_score, biaffine_score])
        # Shape null_score: (batch_size, num_args, num_preds, 1)
        null_score = self.null_score([arg_unary_score, pred_unary_score])
        # Shape final_score: (batch_size, num_args, num_preds, num_labels+1)
        final_score = self.concatenate([total_score, null_score])
        out = final_score
        return out