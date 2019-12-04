import librosa

from feature_extract.feature_extract import extract_feature_vector
from learn_kde_audio import map_factor_learn, emotion_space_map
from recommend import recommend_songs


def main(song_path, train_audio, train_pdfs):
    """Performs all needed actions for the project when a new song is uploaded

    1. Performs feature extraction and dimensionality reduction on the new song
    2. Learns mapping factors from the training audio features and new song audio features
    3. Performs emotion space mapping with training PDFs and mapping factors
    4. Performs recommendation by comparing new PDF to all existing PDFs based on euclidean distance
    """

    signal, sr = librosa.load(str(song_path))

    # 1
    all_features = extract_feature_vector(signal, sr)

    # 2
    map_factor = map_factor_learn(train_audio, all_features)

    # 3 - TODO
    song_pdf = emotion_space_map(train_pdfs, map_factor)

    # 4 - TODO
    recommend = recommend_songs(song_pdf, train_pdfs)




if __name__ == "__main__":
    print(main())
