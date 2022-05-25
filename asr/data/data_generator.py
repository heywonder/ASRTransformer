# coding=utf8
import json
import numpy as np
import pathlib
from tqdm import tqdm
from multiprocessing import Pool
from asr.data.data_processor import generate_spectrogram_features
from utils.wave_util import read_wave_file, dump2json, dump2file


class DataGenerator(object):
    def __init__(self, cfg):
        self.data_dir = cfg['data_dir']
        self.cfg = cfg

    def generate_dataset(self):
        wav_paths = [str(p) for p in pathlib.Path(self.data_dir).glob('**/*.wav')]
        text_paths = [str(p) for p in pathlib.Path(self.data_dir).glob('**/*.txt')]
        wav_infos = self.generate_wav_infos(wav_paths)
        dict_infos, text_infos = self.generate_dict_and_text_infos(text_paths)
        self.save_dataset(wav_infos, text_infos, dict_infos)

    def save_dataset(self, wav_infos, text_infos, dict_infos):
        dataset = {}
        dataset.update(wav_infos)
        for t in dataset:
            for utt in dataset[t]:
                dataset[t][utt].update(text_infos.get(utt, {'invalid': True}))
            dataset[t] = {k: v for k, v in dataset[t].items() if v.get('invalid', False) == False}
            dump2json(list(dataset[t].values()), self.data_dir + f'{t}_dataset.json')
        dump2json(dict_infos, self.data_dir + 'dict.json')

    def generate_wav_infos(self, wav_paths):
        wav_infos = {'train': {}, 'dev': {}, 'test': {}}
        for wav_path in tqdm(wav_paths):
            f = self.generate_features(wav_path)
            wav_type = 'dev' if 'dev' in wav_path else 'test' if 'test' in wav_path else 'train'
            wav_infos[wav_type][f['utt']] = f
        return wav_infos

    def generate_dict_and_text_infos(self, text_paths):
        all_words, text_infos = [], {}
        for text_path in tqdm(text_paths):
            with open(text_path, 'r') as f:
                lines = f.readlines()
                if len(lines) == 1:
                    utt = text_path[:-4].split('/')[-1]
                    words = ['<sos>'] + [i for i in lines[0].replace('\n', '').split(' ')] + ['<eos>']
                    all_words.extend(words)
                    text_infos[utt] = {'utt': utt, 'words': words, 'words_length': len(words)}
                else:
                    for line in lines:
                        utt = line.split(' ')[0]
                        words = ['<sos>'] + [i for i in ''.join(line.replace('\n', '').split(' ')[1:])] + ['<eos>']
                        all_words.extend(words)
                        text_infos[utt] = {'utt': utt, 'words': words, 'words_length': len(words)}
        dict_infos = dict(zip(set(all_words), range(len(set(all_words)))))
        for utt in text_infos:
            text_infos[utt]['word_ids'] = [dict_infos[i] for i in text_infos[utt]['words']]
        dict_infos = {v: k for k, v in dict_infos.items()}
        return dict_infos, text_infos

    def generate_features(self, wav_path):
        wave_data, channel_cnt, samp_width, frame_rate, frame_cnt = read_wave_file(wav_path)
        utt, feature_path = wav_path[:-4].split('/')[-1], wav_path[:-4] + '.feat'
        feature = generate_spectrogram_features(
            wave_data, samplerate=self.cfg['samplerate'], winlen=self.cfg['winlen'],
            winstep=self.cfg['winstep'], nfilt=self.cfg['nfilt'])
        dump2file(feature, feature_path)
        return {'utt': utt, 'feature_path': feature_path, 'feature_shape': feature.shape}
