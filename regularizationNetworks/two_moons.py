import scipy.io as sio
import numpy as np
from flipLabels import fliplabels


def two_moons(npoints, pflip):
    mat_contents = sio.loadmat('./datasets/moons_dataset.mat')
    xtr = mat_contents['Xtr']
    ytr = mat_contents['Ytr']
    xts = mat_contents['Xts']
    yts = mat_contents['Yts']
    npoints = min([100, npoints])
    i = np.random.permutation(100)
    sel = i[0:npoints]
    xtr = xtr[sel, :]
    ytrn = fliplabels(ytr[sel], pflip)
    ytsn = fliplabels(yts, pflip)

    return [xtr, ytrn, xts, ytsn]