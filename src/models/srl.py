from tensorflow.keras.models import Model

from tensorflow.keras.layers import Concatenate, Dropout, Dense, Embedding, Softmax, Input
from models.blocks import Attention, BiHLSTM, SpanEndpointsLength, Scoring, ComputeScoring, CharacterEmbedding, PredicateArgumentEmb
from models.layers import BiAffine
from utils.utils import create_span

class SRL(Model):

    def __init__(self, config, **kwargs):
        super(SRL, self).__init__(**kwargs)
        # Configurations and constants
        self.max_span_width = config['max_span_width']
        self.max_tokens = config['max_tokens']
        self.num_labels = config['num_srl_labels']
        self.max_char = config['max_char']
        span_idx = create_span(self.max_tokens, self.max_span_width)
        self.idx_span_start, self.idx_span_end, self.span_width_list = span_idx 

        # Blocks and Layers
        dropout_val = config['dropout_value']
        self.character_block = CharacterEmbedding(config, name="character_embedding")
        # More embedding layer for wword emb and elmo?
        self.concatenate_1 = Concatenate(name="token_representation")
        self.dropout_1 = Dropout(dropout_val)

        self.bihlstm = BiHLSTM(config, name="BiLSTM")

        # MLP for predicate and argument (questionable)
        self.mlp_pred = Dense(config['mlp_pred_units'], activation='relu')
        self.mlp_arg = Dense(config['mlp_arg_units'], activation='relu')       

        # Span representation for argument 
        self.span_endpoints_len = SpanEndpointsLength(span_idx, name="span_endpoints_and_length")
        self.span_attention = Attention(config, span_idx, name="span_attention_head")
        self.span_length_emb = Embedding(input_dim=self.max_span_width, output_dim=config['span_width_emb'], name='span_width_emb')
        self.concatenate_2 = Concatenate(name='arg_span_representation')

        # Unary score
        self.pred_unary = Scoring(config, 1, name='pred_unary_score')
        self.arg_unary = Scoring(config, 1, name='arg_unary_score')
    
        # Scoring for predicate-argument pair
        self.pred_arg_pair = PredicateArgumentEmb(name="pred_arg_pair")
        self.pred_arg_score = Scoring(config, self.num_labels, name="pred_arg_score")
        self.biaffine_score = BiAffine(self.num_labels, name='biaffine_relation_score')
        self.compute_score = ComputeScoring(self.num_labels, name='compute_final_score')

        self.softmax = Softmax(name="softmax_labels")
    def call(self, inputs): 
        # Shape word_emb: (batch_size, max_tokens, emb)
        # Shape elmo_emb: (batch_size, max_tokens, emb)
        # Shape char_input: (batch_size, max_tokens, max_char)
        word_emb, elmo_emb, char_input = inputs
        
        # Shape char_emb: (batch_size, max_tokens, char_output_dim)
        char_emb = self.character_block(char_input)
        # Shape token_rep: (batch_size, max_tokens, emb)
        token_rep = self.concatenate_1([word_emb, elmo_emb, char_emb])
        token_rep = self.dropout_1(token_rep)
        token_rep = self.bihlstm(token_rep)

        # MLP for predicate and argument
        mlp_pred_out = self.mlp_pred(token_rep) # Shape: (batch_size, max_tokens, mlp_units)
        mlp_arg_out = self.mlp_arg(token_rep) # Shape: (batch_size, max_tokens, mlp_units)

        # Span representation for argument
        # Shape span_start_emb, span_end_emb: (batch_size, num_spans, emb)
        span_start_emb, span_end_emb, span_length = self.span_endpoints_len(mlp_arg_out) 
        span_head_emb = self.span_attention(mlp_arg_out)
        # Shape span_length: (batch_size, num_spans)
        span_width_emb = self.span_length_emb(span_length)
        # Shape arg_rep: (batch_size, num_args, emb)
        arg_rep = self.concatenate_2([span_start_emb, span_end_emb, span_head_emb, span_width_emb])

        # Shape pred_rep: (batch_size, num_preds, emb)
        pred_rep = mlp_pred_out

        # Unary score for predicate and argument
        # Shape pred_unary_score: (batch_size, num_preds, 1)
        pred_unary_score = self.pred_unary(pred_rep)
        # Shape arg_unary_score: (batch_size, num_args, 1)
        arg_unary_score = self.arg_unary(arg_rep)

        # Scoring for predicate-argument pair
        # Shape pred_arg_emb: (batch_size, num_args, num_preds, emb)
        pred_arg_emb = self.pred_arg_pair([arg_rep, pred_rep])
        # Shape pred_arg_emb: (batch_size, num_args, num_preds, num_labels)
        pred_arg_score = self.pred_arg_score(pred_arg_emb)
        relation_score = self.biaffine_score([arg_rep, pred_rep])

        # Shape final_score: (batch_size, num_args, num_preds, num_labels+1) 1= null label
        final_score = self.compute_score([arg_unary_score, pred_unary_score, pred_arg_score, relation_score])
        # Shape out: (batch_size, num_args, num_preds, num_labels+1) 1= null label
        out = self.softmax(final_score)
        return out

    def model(self):
        word_emb = Input(shape=(self.max_tokens, 200))
        elmo_emb = Input(shape=(self.max_tokens, 100))
        char = Input(shape=(self.max_tokens, self.max_char))
        inputs = [word_emb, elmo_emb, char]
        return Model(inputs=inputs, outputs=self.call(inputs))