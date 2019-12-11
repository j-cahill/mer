import numpy as np


def corr_coef(x, y):
    """"Returns Pearson's Correlation Coefficient (R)"""
    return np.corrcoef(x, y)[0, 1]


def recommend_songs(song_pdf, train_pdfs, num_rec=5):
    """ Recommend songs based on Pearson correlation of the PDFs"""
    r = np.apply_along_axis(lambda x: corr_coef(song_pdf, x), axis=1, arr=train_pdfs)
    song_recs = np.argsort(r)[::-1][:num_rec]

    return train_pdfs.iloc[song_recs].index.values
