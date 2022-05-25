# coding=utf8
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from asr.transformer.transformer import Transformer
from utils.mask_utils import get_QK_mask_tensor, get_Z_mask_tensor, get_sequence_mask_tensor


def get_batch_masks(n_samples, n_head, n_rows_encoder, n_rows_decoder, lengths_encoder, lengths_decoder):
    batch_masks = {
        'encoder_QK_mask': get_QK_mask_tensor(n_samples, n_head, n_rows_encoder, n_rows_encoder, lengths_encoder),
        'decoder_QK_mask': get_QK_mask_tensor(
            n_samples, n_head, n_rows_decoder, n_rows_decoder, lengths_decoder, need_seq_mask=True),
        'decoder_QK_mask2': get_QK_mask_tensor(
            n_samples, n_head, n_rows_decoder, n_rows_encoder, lengths_decoder),
        'encoder_Z_mask': get_Z_mask_tensor(n_samples, n_rows_encoder, lengths_encoder),
        'decoder_Z_mask': get_Z_mask_tensor(n_samples, n_rows_decoder, lengths_decoder)
    }
    return batch_masks

class ASRModel(object):
    def __init__(self, args):
        self.args = args
        self.sos_id = args['sos_id']
        self.eos_id = args['eos_id']
        self.transformer = Transformer(
            d_model=self.args['d_model'],
            d_qk=self.args['d_qk'],
            d_v=self.args['d_v'],
            d_ff=self.args['d_ff'],
            n_vocab=self.args['n_vocab'],
            n_encoder=self.args['n_encoder'],
            n_decoder=self.args['n_decoder'],
            n_head=self.args['n_head'],
            max_sequence_len=self.args['max_sequence_len'],
            dropout_rate=self.args['dropout_rate']
        )

    @tf.function
    def _train_one_batch(self, Xs_input, ys_input, ys_true, masks):
        with tf.GradientTape() as tape:
            ys_pred = self.transformer(Xs_input, ys_input, masks=masks, training=True)
            ys_pred = tf.boolean_mask(ys_pred, masks['output_mask'])
            ys_true = tf.boolean_mask(ys_true, masks['output_mask'])
            loss = self.loss_object(ys_true, ys_pred)
        gradients = tape.gradient(target=loss, sources=self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(ys_true, ys_pred)

    @tf.function
    def _test_one_batch(self, Xs_input, ys_input, ys_true, masks):
        ys_pred = self.transformer.model(Xs_input, ys_input, masks=masks, training=False)
        ys_pred = tf.boolean_mask(ys_pred, masks['output_mask'])
        ys_true = tf.boolean_mask(ys_true, masks['output_mask'])
        loss = loss_object(ys_true, ys_pred)

        self.test_loss(loss)
        self.test_accuracy(ys_true, ys_pred)
    
    def _pad_batch_data(self, data_loader, batch_idx):
        Xs_input, ys_input, ys_true, shape_infos = data_loader.get_batch(batch_idx)
        Xs_input = tf.keras.preprocessing.sequence.pad_sequences(Xs_input, value=0, padding='post', dtype='float32')
        ys_input = tf.keras.preprocessing.sequence.pad_sequences(ys_input, value=self.eos_id, padding='post')
        ys_true = tf.keras.preprocessing.sequence.pad_sequences(ys_true, value=-1, padding='post')

        n_samples, n_rows_encoder, n_rows_decoder = Xs_input.shape[0], Xs_input.shape[1], ys_input.shape[1]
        lengths_encoder, lengths_decoder = shape_infos['X_lengths'], shape_infos['y_lengths']
        masks = get_batch_masks(n_samples, self.args['n_head'], n_rows_encoder, n_rows_decoder, lengths_encoder, lengths_decoder)
        masks['output_mask'] = tf.not_equal(ys_input, self.eos_id)
        return Xs_input, ys_input, ys_true, masks

    def train(self, train_data_loader, test_data_loader=None, params={'epochs': 100}):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        need_test = test_data_loader is not None
        if need_test:
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')
            self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        for epoch in range(params['epochs']):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for batch_idx in tqdm(range(train_data_loader.batch_cnt)):
                Xs_input, ys_input, ys_true, masks = self._pad_batch_data(train_data_loader, batch_idx)
                self._train_one_batch(Xs_input, ys_input, ys_true, masks)
            
            if need_test:
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()
                for batch_idx in tqdm(range(test_data_loader.batch_cnt)):
                    Xs_input, ys_input, ys_true, masks = self._pad_batch_data(test_data_loader, batch_idx)
                    self._test_one_batch(Xs_input, ys_input, ys_true, masks)
            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Accuracy: {self.train_accuracy.result() * 100}, '
                f'Test Loss: {test_loss.result() if need_test else None}, '
                f'Test Accuracy: {test_accuracy.result() * 100 if need_test else None}'
            )
            tf.saved_model.save(self.transformer, self.args['model_dir'])

    def beam_predict(self, X_input, beam_size=10, n_best=1, max_output_len=50):
        n_samples, _, _ = X_input.shape
        assert n_samples == 1
        encoder_output = self.transformer.encoder_predict(X_input, masks=None, training=False)

        score_preds = [{'score': 0, 'y_pred': [self.sos_id]}]
        beam_predict_result = []
        for i in tqdm(range(max_output_len + 1)):
            topK_preds = []
            for score_pred in score_preds:
                score, y_pred = score_pred['score'], score_pred['y_pred']
                if y_pred[-1] == self.eos_id or i == max_output_len:
                    beam_predict_result.append(score_pred)
                    if len(beam_predict_result) >= n_best:
                        break
                    continue

                decoder_masks = {
                    'decoder_QK_mask': get_sequence_mask_tensor(n_samples, len(y_pred))
                }
                y_input = np.array([y_pred])
                decoder_output = self.transformer.decoder_predict(encoder_output, y_input, decoder_masks, training=False)
                last_seq_probs = self.transformer.transformer_predict(decoder_output[:, -1, :], output_type='log_softmax')
                topK_probs, topK_idx = tf.math.top_k(last_seq_probs, beam_size)
                topK_probs = topK_probs.numpy()[0]
                topK_idx = topK_idx.numpy()[0]
                for i in range(beam_size):
                    topK_preds.append({'score': score + topK_probs[i], 'y_pred': y_pred + [topK_idx[i]]})
            if len(topK_preds) == 0:
                break
            score_preds = sorted(topK_preds, key=lambda x: x['score'], reverse=True)[:beam_size]

        n_best_predict_result = sorted(beam_predict_result, key=lambda x: x['score'], reverse=True)[:n_best]
        return n_best_predict_result

