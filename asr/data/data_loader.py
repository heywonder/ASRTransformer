# coding=utf8
import numpy as np
import random
from asr.data.data_processor import stack_spectrogram
from utils.wave_util import load_file, load_json


class DataLoader(object):
    def __init__(self, dataset_path, dict_path, batch_size, stack_cnt, if_mask=False):
        self.dataset_path = dataset_path
        self.dict_path = dict_path
        self.batch_size = batch_size
        self.stack_cnt = stack_cnt
        self.if_mask = if_mask
        self._load_dataset()

    def _load_dataset(self):
        self.dataset = load_json(self.dataset_path)
        self.dataset.sort(key=lambda x: x['feature_shape'][0], reverse=True)
        self.id2word = {int(k): v for k, v in load_json(self.dict_path).items()}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.n_vocab = len(self.id2word)
        self.mini_batchs = [self.dataset[i * self.batch_size: (i + 1) * self.batch_size]
                            for i in range(len(self.dataset) // self.batch_size + 1)]
        self.batch_cnt = len(self.mini_batchs)

    def get_batch(self, batch_idx):
        batch = self.mini_batchs[batch_idx]
        spectrograms = self.mask_spectrogram([load_file(sample['feature_path']) for sample in batch])
        Xs_input = self.get_X_input(spectrograms)
        ys_input = [np.array(sample['word_ids'][:-1]) for sample in batch]
        ys_output = [np.array(sample['word_ids'][1:]) for sample in batch]
        shape_infos = {
            'X_lengths': np.array([sample['feature_shape'][0] for sample in batch]),
            'y_lengths': np.array([sample['words_length'] - 1 for sample in batch]),
        }
        return Xs_input, ys_input, ys_output, shape_infos

    def get_X_input(self, spectrograms):
        Xs = []
        for spectrogram in spectrograms:
            X = stack_spectrogram(spectrogram, self.stack_cnt)
            Xs.append(np.vstack(X))
        return Xs

    def mask_spectrogram(self, spectrograms):
        if not self.if_mask:
            return spectrograms

        for i in range(len(spectrograms)):
            mode, h_start, h_width, v_start, v_width = self.get_spec_augment_conditions(spectrograms[i].shape)
            if mode > 90:
                spectrograms[i][h_start: h_start + h_width, v_start: v_start + v_width] = 0
            elif mode > 75:
                spectrograms[i][:, v_start: v_start + v_width] = 0
            elif mode > 60:
                spectrograms[i][h_start: h_start + h_width, :] = 0
        return spectrograms

    def get_spec_augment_conditions(self, shape):
        mode = random.randint(1, 100)
        h_start = random.randint(1, shape[0])
        h_width = random.randint(1, 100)
        v_start = random.randint(1, shape[1])
        v_width = random.randint(1, 100)
        return mode, h_start, h_width, v_start, v_width
