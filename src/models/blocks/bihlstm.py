from tensorflow.keras.layers import Layer, LSTM, Concatenate, Dropout

from ..layers import Highway

class BiHLSTM(Layer):
    def __init__(self, config, name, **kwargs):
        super(BiHLSTM, self).__init__(**kwargs)
        self.forwards = []
        self.backwards = []
        self.highway = []
        self.concatenate = Concatenate(name='lstm_output')
        self.dropout = Dropout(config['lstm_dropout'])
        for i in range(config['lstm_layers']):
            self.forwards.append(LSTM(config['lstm_units'], return_sequences=True, name='lstm_forward_' +name+ str(i)))
            self.backwards.append(LSTM(config['lstm_units'], go_backwards=True, return_sequences=True, name='lstm_backward_'+name+str(i)))
            self.highway.append(Highway(name='highway'+str(i)))
                  
    def call(self, input): # Shape input: (batch_size, max_tokens, emb)
        current_input = input
        for lstm_f, lstm_b, highway in zip(self.forwards, self.backwards, self.highway):
            # Shape f, b: (batch_size, max_tokens, lstm_units)
            f = lstm_f(current_input)
            b = lstm_b(current_input)

            # Shape c: (batch_size, max_tokens, lstm_units*2)
            c = self.concatenate([f,b])
            c = self.dropout(c)
            c = highway(c)

            out = c
            current_input = out

        return out