import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def map_factor_learn(train_audio, test_audio, n_neighbors=4):
    """ Returns mapping factors vector based on 4-NN algorithm

    :argument train - a 2d np array of training data
    :argument test - a single test vector of data to

    :returns: a np array of the mapping factors of test

    STILL NEEDS TO BE TESTED
    """
    scaler = StandardScaler().fit(train_audio)
    train_scale = scaler.transform(train_audio)
    test_scale = scaler.transform(test_audio.reshape(-1, 1).T)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree') \
        .fit(train_scale)

    _, indices = nbrs.kneighbors(test_scale)

    map_fac = np.zeros(train_audio.shape[0])
    map_fac[indices] = 1

    return map_fac


def emotion_space_map(train_pdfs, map_factor):
    """ Output the PDF of a piece of music based on its mapping factors

    :argument train_pdfs: an nx256 matrix of the training pdfs (can be reshaped to 16x16 later if necessary)
    :argument map_factor: an nx1 array of 0's and 1's

    :returns: a 256x1 array which represents an empirical PDF
    """
    new_kde = np.sum(train_pdfs[map_factor.astype(bool)], axis=0)

    # normalize to a pdf
    new_sum = np.sum(new_kde)
    new_pdf = new_kde / new_sum

    return new_pdf


def rescale_pdf_vals(num):
    """ Rescales the empirical pdf to be within the range of -1 to 1

    Assumes a 16x16 pdf
    """

    num_rescale = 2 * (num - 0) / 16 - 1
    return num_rescale


def get_va_vals(pdf):
    """ Get the mean VA values of the PDF
    :argument pdf: a 256x1 numpy array representation of an array
    """
    pdf_square = pdf.reshape(16, 16)
    argmax = np.unravel_index(pdf_square.argmax(), pdf_square.shape)

    arousal, valence = rescale_pdf_vals(argmax[0]), rescale_pdf_vals(argmax[1])

    return arousal, valence
