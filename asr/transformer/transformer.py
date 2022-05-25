# coding=utf8
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model
from tensorflow.nn import softmax, log_softmax
from asr.transformer.encoder import Encoder
from asr.transformer.decoder import Decoder
from asr.transformer.modules import EncoderInput, DecoderInput


class Transformer(Model):
    def __init__(self, d_model, d_qk, d_v, d_ff, n_vocab,
                 n_encoder=6, n_decoder=6, n_head=8, max_sequence_len=5000, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.n_vocab = n_vocab
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.n_head = n_head
        self.max_sequence_len = max_sequence_len
        self.dropout_rate = dropout_rate
        self.encoder_input = EncoderInput(
            d_model=d_model,
            max_sequence_len=max_sequence_len,
            dropout_rate=dropout_rate
        )
        self.decoder_input = DecoderInput(
            n_vocab=n_vocab,
            d_model=d_model,
            max_sequence_len=max_sequence_len,
            dropout_rate=dropout_rate
        )
        self.encoder = Encoder(
            d_model=d_model,
            d_qk=d_qk,
            d_v=d_v,
            d_ff=d_ff,
            n_encoder=n_encoder,
            n_head=n_head,
            dropout_rate=dropout_rate
        )
        self.decoder = Decoder(
            d_model=d_model,
            d_qk=d_qk,
            d_v=d_v,
            d_ff=d_ff,
            n_decoder=n_decoder,
            n_head=n_head,
            dropout_rate=dropout_rate
        )
        self.dim = Dense(n_vocab)

    def encoder_predict(self, X_input, masks=None, training=None):
        if masks is None:
            masks = {}
        encoder_input = self.encoder_input(X_input, training=training)
        encoder_QK_mask, encoder_Z_mask = masks.get('encoder_QK_mask'), masks.get('encoder_Z_mask')
        encoder_output = self.encoder(encoder_input, encoder_QK_mask, encoder_Z_mask, training=training)
        return encoder_output

    def decoder_predict(self, encoder_output, y_input, masks=None, training=None):
        if masks is None:
            masks = {}
        decoder_input = self.decoder_input(y_input, training=training)
        decoder_QK_mask, decoder_Z_mask = masks.get('decoder_QK_mask'), masks.get('decoder_Z_mask')
        decoder_QK_mask2 = masks.get('decoder_QK_mask2')
        decoder_output = self.decoder(
            decoder_input, encoder_output, decoder_QK_mask, decoder_QK_mask2, decoder_Z_mask, training=training)
        return decoder_output
    
    def transformer_predict(self, decoder_output, output_type=None):
        transformer_output = self.dim(decoder_output)
        if output_type == 'softmax':
            transformer_output = softmax(transformer_output)
        elif output_type == 'log_softmax':
            transformer_output = log_softmax(transformer_output)
        return transformer_output
    def call(self, X_input, y_input, masks=None, training=None):
        encoder_output = self.encoder_predict(X_input, masks=masks, training=training)
        decoder_output = self.decoder_predict(encoder_output, y_input, masks=masks, training=training)
        transformer_output = self.dim(decoder_output)
        return transformer_output

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_qk': self.d_qk,
            'd_v': self.d_v,
            'n_vocab': self.n_vocab,
            'n_encoder': self.n_encoder,
            'n_decoder': self.n_decoder,
            'n_head': self.n_head,
            'max_sequence_len': self.max_sequence_len,
            'transformer_dropout_rate': self.dropout_rate,
        })
        return config

