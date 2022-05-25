# coding=utf8
from tensorflow.keras.layers import Layer, LayerNormalization
from asr.transformer.modules import MultiHeadAttention, FeedForward


class Decoder(Layer):
    def __init__(self, d_model, d_qk, d_v, d_ff, n_decoder=6, n_head=8, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.n_decoder = n_decoder  # Encoder模块包含encoder个数，默认6个
        self.n_head = n_head  # multi_head_attention中head个数
        self.d_model = d_model  # 输入的embedding后维度
        self.d_qk = d_qk  # Q、K矩阵维度
        self.d_v = d_v  # V矩阵维度
        self.d_ff = d_ff  # feed_forward中W1参数维度
        self.dropout_rate = dropout_rate

        self.decoder_list = [
            DecoderBlock(
                n_head=self.n_head,
                d_model=self.d_model,
                d_qk=self.d_qk,
                d_v=self.d_v,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            ) for i in range(self.n_decoder)
        ]

    def call(self, decoder_input, encoder_output, QK_mask=None, QK_mask2=None, Z_mask=None, training=None):
        decoder_output = decoder_input
        for decoder in self.decoder_list:
            decoder_output = decoder(decoder_output, encoder_output, QK_mask, QK_mask2, Z_mask, training=training)
        return decoder_output

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'n_encoder': self.n_encoder,
            'n_head': self.n_head,
            'd_model': self.d_model,
            'd_qk': self.d_v,
            'd_ff': self.d_qk,
            'd_v': self.d_ff,
            'decoder_dropout_rate': self.dropout_rate
        })
        return config


class DecoderBlock(Layer):
    def __init__(self, n_head, d_model, d_qk, d_v, d_ff, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.multi_head_attention1 = MultiHeadAttention(
            n_head=self.n_head,
            d_model=self.d_model,
            d_qk=self.d_qk,
            d_v=self.d_v,
            dropout_rate=self.dropout_rate
        )
        self.multi_head_attention2 = MultiHeadAttention(
            n_head=self.n_head,
            d_model=self.d_model,
            d_qk=self.d_qk,
            d_v=self.d_v,
            dropout_rate=self.dropout_rate
        )
        self.feed_forward = FeedForward(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate
        )
        self.norm = LayerNormalization(axis=-1)

    def call(self, decoder_input, encoder_output, QK_mask=None, QK_mask2=None, Z_mask=None, training=None):
        multi_head_attention1 = self.multi_head_attention1(decoder_input, decoder_input, decoder_input, mask=QK_mask, training=training)
        multi_head_attention1 = self.norm(multi_head_attention1 + decoder_input)

        multi_head_attention2 = self.multi_head_attention2(
            query=multi_head_attention1,
            key=encoder_output, 
            value=encoder_output,
            mask=QK_mask2, training=training
        )
        multi_head_attention2 = self.norm(multi_head_attention2 + multi_head_attention1)

        decoder_block_output = self.feed_forward(multi_head_attention2, training=training)
        decoder_block_output = self.norm(decoder_block_output + multi_head_attention2)
        if Z_mask is not None:
            decoder_block_output = decoder_block_output * Z_mask
        return decoder_block_output

    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({
            'n_head': self.n_head,
            'd_model': self.d_model,
            'd_qk': self.d_v,
            'd_ff': self.d_qk,
            'd_v': self.d_ff,
            'decoder_block_dropout_rate': self.dropout_rate
        })
        return config

