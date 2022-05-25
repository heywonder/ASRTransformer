# coding=utf8
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Embedding, Softmax, Dense, LayerNormalization
from tensorflow.keras.initializers import glorot_uniform, zeros


class PositionEncoding(Layer):
    def __init__(self, d_model, max_sequence_len):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_sequence_len = max_sequence_len
        self._generate_position_encoding()
    
    def _generate_position_encoding(self):
        all_position_encoding = np.array([
            [pos / np.power(10000, 2.0 * i / self.d_model) for i in range(self.d_model)]
            for pos in range(self.max_sequence_len)])
        all_position_encoding[:, 0::2] = np.sin(all_position_encoding[:, 0::2])  # dim 2i
        all_position_encoding[:, 1::2] = np.cos(all_position_encoding[:, 1::2])  # dim 2i+1
        all_position_encoding = tf.cast(all_position_encoding, dtype=tf.float32)
        self.all_position_encoding = all_position_encoding
    
    def call(self, n_samples, n_seqs):
        position_idx = tf.tile(tf.expand_dims(tf.range(n_seqs), 0), [n_samples, 1])
        position_encoding = tf.nn.embedding_lookup(self.all_position_encoding, position_idx)
        return position_encoding

    def get_config(self):  
        config = super(PositionEncoding, self).get_config()
        config.update({
            'd_model': self.d_model,
            'max_sequence_len': max_sequence_len,
        })
        return config


class EncoderInput(Layer):
    def __init__(self, d_model, max_sequence_len=5000, dropout_rate=0.1):
        super(EncoderInput, self).__init__()
        self.d_model = d_model
        self.max_sequence_len = max_sequence_len
        self.dropout_rate = dropout_rate

        self.dim = Dense(d_model)
        self.norm = LayerNormalization(axis=-1)
        self.position_encoding = PositionEncoding(d_model=d_model, max_sequence_len=max_sequence_len) 
        self.dropout = Dropout(dropout_rate)
        
    def call(self, X, training=None):
        n_samples, n_seqs = tf.shape(X)[0], tf.shape(X)[1]
        dim_and_norm = self.norm(self.dim(X))
        position_encoding = self.position_encoding(n_samples, n_seqs)
        encoder_input = self.dropout(dim_and_norm + position_encoding, training=training)
        return encoder_input

    def get_config(self):
        config = super(EncoderInput, self).get_config()
        config.update({
            'd_model': self.d_model,
            'max_sequence_len': self.max_sequence_len,
            'encoder_input_dropout_rate': self.dropout_rate,
        })
        return config


class DecoderInput(Layer):
    def __init__(self, n_vocab, d_model, max_sequence_len=5000, dropout_rate=0.1):
        super(DecoderInput, self).__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.max_sequence_len = max_sequence_len
        self.dropout_rate = dropout_rate

        self.embedding = Embedding(
            input_dim=n_vocab+1,
            output_dim=d_model,
            embeddings_initializer=glorot_uniform
        )
        self.position_encoding = PositionEncoding(d_model=d_model, max_sequence_len=max_sequence_len) 
        self.dropout = Dropout(dropout_rate)
        
    def call(self, X, training=None):
        n_samples, n_seqs = tf.shape(X)[0], tf.shape(X)[1]
        embedding = self.embedding(X)
        position_encoding = self.position_encoding(n_samples, n_seqs)
        decoder_input = self.dropout(embedding + position_encoding, training=training)
        return decoder_input

    def get_config(self):
        config = super(EncoderInput, self).get_config()
        config.update({
            'n_vocab': self.n_vocab,
            'd_model': self.d_model,
            'max_sequence_len': self.max_sequence_len,
            'decoder_input_dropout_rate': self.dropout_rate,
        })
        return config


class MultiHeadAttention(Layer):
    def __init__(self, n_head, d_model, d_qk, d_v, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_qk = d_qk
        self.d_v = d_v
        self.dropout_rate = dropout_rate

        self.WQ = Dense(self.n_head * self.d_qk)
        self.WK = Dense(self.n_head * self.d_qk)
        self.WV = Dense(self.n_head * self.d_v)
        self.dim = Dense(self.d_model)
        self.softmax = Softmax(axis=-1)
        self.dropout = Dropout(self.dropout_rate)

    def call(self, query, key, value, mask=None, training=None):
        n_samples, n_seqs_q, n_seqs_kv = tf.shape(query)[0], tf.shape(query)[1], tf.shape(key)[1]
        
        Q, K, V = self.WQ(query), self.WK(key), self.WV(value)
        Q = tf.reshape(tf.concat(tf.split(Q, self.n_head, axis=-1), axis=1), (n_samples, self.n_head, n_seqs_q, self.d_qk))
        K = tf.reshape(tf.concat(tf.split(K, self.n_head, axis=-1), axis=1), (n_samples, self.n_head, n_seqs_kv, self.d_qk))
        V = tf.reshape(tf.concat(tf.split(V, self.n_head, axis=-1), axis=1), (n_samples, self.n_head, n_seqs_kv, self.d_qk))

        QK = tf.matmul(Q, tf.transpose(K, perm=[0, 1, 3, 2])) / tf.pow(tf.cast(self.d_qk, tf.float32), 0.5)
        if mask is not None:  # 需要mask的对应的位置值为-inf，其他位置值为0
            QK = QK + mask

        attention = self.dropout(self.softmax(QK), training=training)
        Z = tf.matmul(attention, V)
        Z = tf.reshape(tf.concat(tf.split(Z, self.n_head, axis=1), axis=-1), (n_samples, n_seqs_q, self.n_head * self.d_v))
        multi_head_attention = self.dropout(self.dim(Z), training=training)
        return multi_head_attention

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'n_vocab': self.n_head,
            'd_model': self.d_model,
            'd_qk': self.d_qk,
            'd_v': self.d_model,
            'multihead_attention_dropout_rate': self.dropout_rate,
        })
        return config


class FeedForward(Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.dim1 = Dense(self.d_ff, activation='relu')
        self.dim2 = Dense(self.d_model)
        self.dropout = Dropout(self.dropout_rate)

    def call(self, X, training=None):
        ff_in = self.dim1(X)
        ff_out = self.dim2(ff_in)
        self.dropout(ff_out, training=training)
        return ff_out

    def get_config(self):
        config = super(FeedForward, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff
        })
        return config
    

