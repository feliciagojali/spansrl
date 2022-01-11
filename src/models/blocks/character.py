from tensorflow.keras.layers import Layer, TimeDistributed, Conv1D, Embedding, Dropout, MaxPooling1D, Flatten

class CharacterEmbedding(Layer):
    def __init__(self, config, **kwargs):
        super(CharacterEmbedding, self).__init__(**kwargs)
        self.embedding = TimeDistributed(Embedding(config['char_vocab'], config['char_output_dim']), name='char_embedding')
        self.dropout_1 = Dropout(config['char_dropout'])
        self.conv1d_out = TimeDistributed(Conv1D(kernel_size=config['char_cnn_kernel_size'], filters=config['char_output_dim'], padding='same',activation='tanh', strides=1))
        self.maxpool_out = TimeDistributed(MaxPooling1D(config['max_char']))
        self.flatten = TimeDistributed(Flatten())
        self.dropout_2 = Dropout(config['char_dropout'])

    def call(self, input): # Shape input: (batch_size, max_tokens, max_char)
        # Shape x: (batch_size, max_tokens, max_char, char_output_dim)
        x = self.embedding(input)
        x = self.dropout_1(x)
        x = self.conv1d_out(x)
        # Shape m: (batch_size, max_tokens, 1, char_output_dim)
        m = self.maxpool_out(x)
        # Shape f: (batch_size, max_tokens, char_output_dim)
        f = self.flatten(m)
        f = self.dropout_2(f)

        out = f
        return out