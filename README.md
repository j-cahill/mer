# mer

mer (Music Emotion Recognition) is a python package that makes music recommendations based on the emotion elicited by a song. (Currently still in progress)

[![Build Status](https://travis-ci.com/j-cahill/mer.svg?branch=master)](https://travis-ci.com/j-cahill/mer)



The package is based on the paper [Predicting the Probability Density Function of
Music Emotion Using Emotion Space Mapping](<https://ieeexplore.ieee.org/document/7745959>) The goal of this package is to create recommendations for new music based on the emotion elicited by some training music.

The process consists of

1.  Perform KDE on each of our training songs based on [valence and arousal](<https://cxl.com/blog/valence-arousal-and-how-to-kindle-an-emotional-fire/>) data and create mxm matrices of empirical probability distributions for each.
2. We extract the audio features of each of our training songs based on their clips.
3. For each new song that comes in
   1. We extract all of its audio features
   2. We perform *mapping factor learning*  based on our dictionary of known training song audio features (will be pulled from database) and return a mapping factors vector
   3. Given the mapping factors vector from 3.2, we perform *emotion space mapping* based on our dictionary of known KDEs (will be pulled from database) and return a new KDE
   4. We perform recommendations based on the songs that have the smallest KL Divergence with the input song.



## Usage

Note: as of right now the package doesn't contain code to perform KDE on mp3 files and create `train_pdfs`, so its necessary to have the KDE performed already.  I'm working on adding this functionality moving forward.

```python
from mer.mer import mer, make_train

# First we do audio feature extraction on our training data
audio_train = 	make_train(train_audio_dir, song_ids)

# Now we perform recommendations based on a new song
recs = main(song_path, audio_train, train_pdfs, num_recs=5)
```

