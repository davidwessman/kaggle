import os
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile as wav
import scipy.signal as signal

X_SIZE = 128000
IMG_SIZE = 256


def signal_spectrogram(filepath):
    sample_rate, samples = wav.read(filepath)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    return spectrogram


def spectrogram(filepath):
    sample_rate, samples = wav.read(filepath)

    window_length = 512
    window_shift = 121

    if len(samples) > X_SIZE:
        print(len(samples))
        samples = samples[:X_SIZE]

    X = np.zeros(X_SIZE).astype('float32')
    X[:len(samples)] += samples
    spec = np.zeros((IMG_SIZE, IMG_SIZE)).astype('float32')

    for i in range(IMG_SIZE):
        start = i * window_shift
        end = start + window_length
        sig = np.abs(np.fft.rfft(X[start:end] * np.hanning(window_length)))
        spec[:, i] = (sig[1:IMG_SIZE + 1])[::-1]

    spec = (spec-spec.min())/(spec.max()-spec.min())
    spec = np.log10((spec * 100 + 0.01))
    spec = (spec-spec.min())/(spec.max()-spec.min()) - 0.5

    return spec

def classes(data):
  found_classes = set()
  for row in data:
    for det_class in row[1]:
      found_classes.add(det_class)
  return found_classes


def main():
    with open('./data/train_curated.csv', 'r') as files:
        data = []
        reader = csv.reader(files)
        _headers = next(reader)
        i = 0
        for row in reader:
            spectro = signal_spectrogram(
                os.path.join('./data/train_curated', row[0]))
            data.append([spectro, list(row[1].split(','))])
            if i > 30:
              break
            i += 1

    return data


# if __name__ == '__main__':
#     main()
