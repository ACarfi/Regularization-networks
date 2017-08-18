import numpy as np


def fliplabels(y, p):
    '''
    % function [Yn]=flipLabels(Y,p)
    % flips p percent of labels to be flipped
    % the labels must be +1 and -1
    '''
    n = y.shape[0]
    n_flips = np.floor(n*p)
    i = np.random.permutation(n)
    sel = i[0:n_flips]
    y[sel] = -1*y[sel]
    return y