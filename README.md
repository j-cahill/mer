- \__init__.py

- setup.py

- README.md

- .gitignore

- requirements.txt

- **libs**

  - \__init__.py
  - **rp_extract**

- **feature_extract**

  - \__init__.py
  - harmonic.py
  - spectral.py
  - temporal.py
  - rhythmic.py
  - extract_all.py

- estimate_kde.py (estimates KDE based on VA data, used for training data only)

- map_factor_learn.py (Learns mapping factors based on training data and audio features)

- emo_space_map.py (Learns KDE based on mapping factor and training KDE's)


Steps this package needs to help us accomplish

1. We perform KDE on each of our training songs based on VA data and create mxm matrices of probability distributions for each
2. We extract the audio features of each of our training songs based on their clips.
3. For each new song that comes in
   1. We extract all of its audio features
   2. We perform *mapping factor learning*  based on our dictionary of known training song audio features (will be pulled from database) and **return a mapping factors vector**
   3. Given the mapping factors vector from 3.2, we perform *emotion space mapping* based on our dictionary of known KDEs (will be pulled from database) and **return a new KDE**