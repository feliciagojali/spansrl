import tensorflow as tf
from tensorflow.keras.layers import Layer, TimeDistributed, Conv1D, Embedding, Dropout, MaxPooling1D, Flatten

class CharacterEmbedding(Layer):
    def __init__(self, config, **kwargs):
        super(CharacterEmbedding, self).__init__(**kwargs)
        self.embedding = TimeDistributed(Embedding(config['char_vocab'], config['char_output_dim']), name='char_embedding')
        self.dropout_1 = Dropout(config['emb_dropout'])
        self.conv1d_out = [TimeDistributed(Conv1D(kernel_size=kernel_sz, filters=config['char_filter_size'], padding='same',activation='relu', strides=1)) for kernel_sz in config['char_cnn_kernel_size']]    
        self.maxpool_out = [TimeDistributed(MaxPooling1D(config['max_char'])) for _ in config['char_cnn_kernel_size']]
        self.flatten = [TimeDistributed(Flatten()) for _ in config['char_cnn_kernel_size']]
        self.dropout_2 = [Dropout(config['emb_dropout']) for _ in config['char_cnn_kernel_size']]

    def call(self, input): # Shape input: (batch_size, max_tokens, max_char)
        # Shape x: (batch_size, max_tokens, max_char, char_output_dim)
        x = self.embedding(input)
        x = self.dropout_1(x)
        # Shape c: (len(kernel_size), batch_size, max_tokens, max_char, char_output_dim)
        c = [conv1d_out(x) for conv1d_out in self.conv1d_out]
        # Shape m: (len(kernel_size), batch_size, max_tokens, 1, char_output_dim)
        m = [maxpool_out(x) for maxpool_out, x in zip(self.maxpool_out, c)]
        # Shape f: (len(kernel_size), batch_size, max_tokens, char_output_dim)
        f = [flatten(x) for flatten, x in zip(self.flatten, m)]
        f = [dropout_2(x) for dropout_2, x in zip(self.dropout_2, f)]
        # Shape out: (batch_size, max_tokens, char_output_dim)
        out = tf.concat([x for x in f], -1)
        return out