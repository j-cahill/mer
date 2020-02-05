import pytest

import mer.feature_extract.rhythmic

# @pytest.fixture
# def song_data(scope='module'):
#     import pathlib
#     import librosa
#
#     data_path = Pathlib
#     return librosa.load(str(path))


def test_extract_rhythmic_features():
    from pathlib import Path
    print('hi')
    print(Path(__file__) / 'tests' / 'data')
    assert 1 == 1

if __name__ == "__main__":
    from pathlib import Path
    print(Path(__file__).parent)
