import numpy as np


def squaredist(x1, x2):
    n = x1.shape[0]
    m = x2.shape[0]

    sq1 = np.sum(np.power(x1, 2), axis=1)
    sq1 = np.reshape(sq1, (n, 1))

    sq2 = np.sum(np.power(x2, 2), axis=1)
    sq2 = np.reshape(sq2, (m, 1))

    d = np.dot(sq1, np.ones([1, m])) + np.dot(np.ones([n, 1]), np.transpose(sq2)) - 2*(np.dot(x1, np.transpose(x2)))

    return d
