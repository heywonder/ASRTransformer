# coding=utf8
from tensorflow.keras.layers import Layer, LayerNormalization
from asr.transformer.modules import MultiHeadAttention, FeedForward


class Encoder(Layer):
    def __init__(self, d_model, d_qk, d_v, d_ff, n_encoder=6, n_head=8, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.n_encoder = n_encoder  # Encoder模块包含encoder个数，默认6个
        self.n_head = n_head  # multi_head_attention中head个数
        self.d_model = d_model  # 输入的embedding后维度
        self.d_qk = d_qk  # Q、K矩阵维度
        self.d_v = d_v  # V矩阵维度
        self.d_ff = d_ff  # feed_forward中W1参数维度
        self.dropout_rate = dropout_rate

        self.encoder_list = [
            EncoderBlock(
                n_head=self.n_head,
                d_model=self.d_model,
                d_qk=self.d_qk,
                d_v=self.d_v,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate
            ) for i in range(self.n_encoder)
        ]

    def call(self, encoder_input, QK_mask=None, Z_mask=None, training=None):
        encoder_output = encoder_input
        for encoder in self.encoder_list:
            encoder_output = encoder(encoder_output, QK_mask, Z_mask, training=training)
        return encoder_output

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'n_encoder': self.n_encoder,
            'n_head': self.n_head,
            'd_model': self.d_model,
            'd_qk': self.d_v,
            'd_ff': self.d_qk,
            'd_v': self.d_ff,
            'encoder_dropout_rate': self.dropout_rate
        })
        return config

class EncoderBlock(Layer):
    def __init__(self, n_head, d_model, d_qk, d_v, d_ff, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(
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

    def call(self, encoder_input, QK_mask=None, Z_mask=None, training=None):
        multi_head_attention = self.multi_head_attention(encoder_input, encoder_input, encoder_input, mask=QK_mask, training=training)
        multi_head_attention = self.norm(multi_head_attention + encoder_input)
        encoder_block_output = self.feed_forward(multi_head_attention, training=training)
        encoder_block_output = self.norm(encoder_block_output + multi_head_attention)
        if Z_mask is not None:
            encoder_block_output = encoder_block_output * Z_mask
        return encoder_block_output

    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({
            'n_head': self.n_head,
            'd_model': self.d_model,
            'd_qk': self.d_v,
            'd_ff': self.d_qk,
            'd_v': self.d_ff,
            'encoder_block_dropout_rate': self.dropout_rate
        })
        return config
    