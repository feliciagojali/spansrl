import tensorflow as tf
from tensorflow.keras.layers import Layer
tf.random.set_seed(42)
# Adopted from https://github.com/PaddlePaddle/PaddleNLP/tree/develop

class BiAffine(Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, num_outputs, bias_x=True, bias_y=True, name='BiAffine'):
        super(BiAffine, self).__init__(name=name)
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.num_outputs = num_outputs

    def build(self, input_shape):
        dim_x = input_shape[0][-1]
        dim_y = input_shape[1][-1]
        self.kernel = self.add_weight("kernel",
                                      shape=[self.num_outputs, dim_x + self.bias_x, dim_y + self.bias_y])
        super(BiAffine, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input):
        x, y = input
        if self.bias_x:
            x = tf.concat([x, tf.ones_like(x[:, :, :1])], axis=-1)
        if self.bias_y:
            y = tf.concat([y, tf.ones_like(y[:, :, :1])], axis=-1)
        
        # Shape x: (batch_size, num_tokens, input_size + bias_x)
        o = self.kernel.shape[0]
        # Shape x: (batch_size, output_size, num_tokens_x, input_size_x + bias_x)
        x = tf.tile(tf.expand_dims(x, axis=1), [1, o, 1, 1])
        # Shape y: (batch_size, output_size, num_tokens_y, input_size_y + bias_y)
        y = tf.tile(tf.expand_dims(y, axis=1), [1, o, 1, 1])
        # Shape a: (batch_size, output_size, num_tokens_x , input_size_y + bias_y)
        a = tf.linalg.matmul(x, self.kernel)
        # Shape s: (batch_size, output_size, num_tokens_x, num_tokens_y)
        s = tf.linalg.matmul(a,  tf.transpose(y, perm=[0, 1, 3, 2]))
        # Shape out: (batch_size, num_tokens_x, num_tokens_y, output_size)
        out = tf.transpose(s, perm=[0, 2, 3, 1])
        return out