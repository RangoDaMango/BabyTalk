import librosa
import numpy as np

def extract_features(file):

    y, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    pitch = librosa.yin(y, fmin=50, fmax=600)
    rms = librosa.feature.rms(y=y)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        [np.mean(pitch)],
        [np.mean(rms)]
    ])

    return features
