# coding=utf8
import numpy as np
import tensorflow as tf


def get_QK_mask_tensor(n_samples, n_head, n_rows, n_cols, lengths, need_seq_mask=False):
    QK_mask = np.zeros(shape=(n_samples, n_rows, n_cols))
    for i in range(len(lengths)):
        QK_mask[i, :, lengths[i]:] = -np.inf
    QK_mask = tf.cast(QK_mask, tf.float32)
    if need_seq_mask:
        QK_mask = get_sequence_mask_tensor(n_samples, n_rows) + QK_mask
    QK_mask = tf.tile(tf.expand_dims(QK_mask, axis=1), (1, n_head, 1, 1))
    return QK_mask

def get_Z_mask_tensor(n_samples, n_rows, lengths):
    Z_mask = np.ones(shape=(n_samples, n_rows, 1))
    for i in range(len(lengths)):
        Z_mask[i, lengths[i]:, :] = 0
    Z_mask = tf.cast(Z_mask, tf.float32)
    return Z_mask

def get_sequence_mask_tensor(n_samples, n_rows):
    sequence_mask = np.zeros(shape=(n_samples, n_rows, n_rows))
    for i in range(n_rows):
        sequence_mask[:, i, i+1:] = -np.inf
    sequence_mask = tf.cast(sequence_mask, tf.float32)
    return sequence_mask


