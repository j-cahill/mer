import numpy as np
import librosa


def append_val(d, k, v):
    if k not in d:
        d[k] = [v]
    else:
        d[k].append(v)
    return d


def salient_pitch(y, sr):
    per_sample_t = 1.0 / sr
    pitch_frame_length, pitch_hop_length = int(50 / 1000 / per_sample_t), int(25 / 1000 / per_sample_t)

    # we hope to introduce an acf - filtered salient pitch
    salient_pitch = []
    num_total_frames = (len(y) - pitch_frame_length) // pitch_hop_length + 1
    for i in range(num_total_frames):
        acf = librosa.core.autocorrelate(y[i * pitch_hop_length: (i * pitch_hop_length + pitch_frame_length)])
        # introduce a filtered acf method
        for j in range(1, len(acf) - 1):
            if acf[j] > acf[j - 1] and acf[j] > acf[j + 1]:
                if j >= 20 and acf[j] == np.max(acf[j - 20:j + 20]):  # reducing the adjacent HF fluctuation
                    salient_pitch.append(sr / j)
                    break
    return np.array(salient_pitch, dtype=float)


def chroma_centroid(y, sr):
    per_sample_t = 1.0 / sr
    chromagram_frame_length, chromagram_hop_length = int(100 / 1000 / per_sample_t), int(12.5 / 1000 / per_sample_t)
    chromagram_center = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=chromagram_frame_length, \
                                                          hop_length=chromagram_hop_length)[0]

    return chromagram_center[1:]


def extract_harmonic_features(signal, sr):

    salient_pitches = salient_pitch(signal, sr)
    salient_pitch_features = np.array([np.mean(salient_pitches), np.std(salient_pitches)])

    chroma_centroids = chroma_centroid(signal, sr)
    chroma_centroid_features = np.array([np.mean(chroma_centroids), np.std(chroma_centroids)])

    all_features = np.concatenate([salient_pitch_features, chroma_centroid_features])

    return all_features
