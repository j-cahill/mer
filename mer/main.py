import warnings
from pathlib import Path

import librosa
import tqdm
import numpy as np
import pandas as pd

from mer.feature_extract.feature_extract import extract_feature_vector
from mer.learn_kde_audio import map_factor_learn, emotion_space_map
from mer.recommend import recommend_songs


def main(song_path, train_audio, train_pdfs, num_rec=5):
    """Performs all needed actions for the project when a new song is uploaded

    1. Performs feature extraction and dimensionality reduction on the new song
    2. Learns mapping factors from the training audio features and new song audio features
    3. Performs emotion space mapping with training PDFs and mapping factors
    4. Performs recommendation by comparing new PDF to all existing PDFs based on euclidean distance
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        signal, sr = librosa.load(str(song_path))

    # 1
    all_features = extract_feature_vector(signal, sr)

    # 2
    map_factor = map_factor_learn(train_audio, all_features)

    # 3
    song_pdf = emotion_space_map(train_pdfs.values, map_factor)

    # 4
    recommend = recommend_songs(song_pdf, train_pdfs, num_rec=num_rec)

    return recommend


def make_train(train_audio_dir):
    """Make the training data for our existing audio"""
    all_mp3_paths = list(train_audio_dir.glob('**/*.mp3'))[:50]  # just 50 for now to test

    audio_train = []
    song_id = []
    print('Creating the audio training dataset:\n')
    for path in tqdm.tqdm(all_mp3_paths):
        song_id.append(path.name[:-4])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            signal, sr = librosa.load(str(path))
        song_features = extract_feature_vector(signal, sr)
        audio_train.append(song_features)

    audio_train = pd.DataFrame(np.array(audio_train), index=song_id)

    return audio_train


if __name__ == "__main__":
    song_path = Path.cwd().parent / 'data' / 'raw' / 'clips_45seconds' / '669.mp3'
    train_audio = pd.read_csv(Path.cwd().parent / 'data' / 'interim' / 'audio_feature_sample_50.csv', index_col=0)
    train_pdfs = pd.read_csv(Path.cwd().parent / 'data' / 'final' / 'Time_Average_Gamma_0_1.csv',
                             index_col='song_id')

    print(main(song_path, train_audio.values, train_pdfs.iloc[:50, :]))
    # audio_train = make_train(song_path.parent)
    # audio_train.to_csv(Path.cwd() / 'data' / 'interim' / 'audio_feature_sample_50.csv')
