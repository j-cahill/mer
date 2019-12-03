from pathlib import Path
import librosa

from feature_extract import temporal


def main():
    test_song = Path.cwd() / 'data' / 'raw' / 'clips_45seconds' / '69.mp3'
    signal, sr = librosa.load(str(test_song))

    temporal_features = temporal.extract_temporal_features(signal, sr)

    return temporal_features


if __name__ == "__main__":
    print(main())