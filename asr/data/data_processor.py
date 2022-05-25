# coding=utf8
import numpy as np
from python_speech_features import logfbank


def generate_spectrogram_features(wave_data, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=80):
    fbank_features = logfbank(
        wave_data, samplerate=samplerate, winlen=winlen, winstep=winstep, nfilt=nfilt)
    return fbank_features

def stack_spectrogram(spectrogram, stack_cnt=3):
    X = []
    spect_length = len(spectrogram)
    for i in range(spect_length):
        if i <= spect_length - stack_cnt:
            frame = np.hstack(spectrogram[i: i + stack_cnt])
        else:
            frame = np.hstack(spectrogram[i:])
            for _ in range(i + stack_cnt - spect_length):
                frame = np.hstack([frame, spectrogram[-1]])
        X.append(frame)
    return X
    