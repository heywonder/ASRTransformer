# coding=utf8
"""
音频文件的读取和处理
"""
import json
import numpy as np
import wave


def read_wave_file(file_name):
    """
    读取音频文件，返回帧数、声道数、帧速率、帧数据等
    """
    wav = wave.open(file_name, "rb")
    channel_cnt, samp_width, frame_rate, frame_cnt, _, _ = wav.getparams()
    wave_data = np.fromstring(wav.readframes(frame_cnt), dtype=np.short)
    wave_data.shape = -1, channel_cnt
    wave_data = wave_data.T
    wav.close()
    return wave_data, channel_cnt, samp_width, frame_rate, frame_cnt


def read_dataset_file(file_name):
    dataset = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            dataset.append(line.split(' '))
    return dataset


def dump2file(feature, path_name):
    with open(path_name, 'wb') as f:
        np.save(f, feature)


def load_file(path_name):
    return np.load(path_name)


def dump2json(data, path_name):
    with open(path_name, 'w') as f:
        json.dump(data, f)


def load_json(path_name):
    with open(path_name, 'r') as f:
        return json.load(f)
