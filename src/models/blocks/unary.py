from tensorflow.keras.layers import Layer, Dense, Dropout

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