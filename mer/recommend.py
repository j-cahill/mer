import numpy as np


def corr_coef(x, y):
    """"Returns Pearson's Correlation Coefficient (R)"""
    return np.corrcoef(x, y)[0, 1]

def kl_divergence(x, y):
    """return kullback divergence, default 16 * 16"""
    return np.sum(x * np.log(np.divide(x, y)) )

def recommend_songs(song_pdf, train_pdfs, num_rec=5, metric='kl'):
    """ Recommend songs based on Pearson correlation of the PDFs"""
    if metric !='kl':
        r = np.apply_along_axis(lambda x: corr_coef(song_pdf, x), axis=1, arr=train_pdfs)
    else:
        r = np.apply_along_axis(lambda x: kl_divergence(song_pdf, x), axis=1, arr=train_pdfs)

    song_recs = np.argsort(r)[::-1][:num_rec]

    return train_pdfs.iloc[song_recs].index.values
