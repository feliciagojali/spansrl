from tensorflow.keras.models import Model

from src.utils.utils import create_span

class SRL(Model):

    def __init__(self, config, **kwargs):
        super(SRL, self).__init__()
        self.max_span_width = config['max_span_width']
        self.max_tokens = config['max_tokens']
        idx_span_start, idx_span_end, span_width = create_span(self.max_tokens, self.max_span_width)
        self.idx_span_start = idx_span_start
        self.idx_span_end = idx_span_end
        self.span_width_list = span_width

    def call(self, inputs): 
        # Shape word_emb: (batch_size, max_tokens, emb)
        # Shape elmo_emb: (batch_size, max_tokens, emb)
        # Shape char_input: (batch_size, max_tokens, max_char)
        word_emb, elmo_emb, char_input = inputs
        
        return self.classifier(x)
