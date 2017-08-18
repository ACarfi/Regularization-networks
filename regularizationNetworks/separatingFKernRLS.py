import numpy as np
from regularizedKernLSTest import regularizedkernlstest
from plotDataSet import plotdataset
import matplotlib.pyplot as plt


def separatingfkernrls(c, xtr, ytr, kernel, sigma, xts):

    '''
    The function classifies points evenly sampled in a visualization area,
    according to the classifier Regularized Least Squares
    
    Inputs:
    c: model weights
    xtr: training input
    ytr: training output
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    sigma: width of the gaussian kernel, if used
    xts: test points
    
    Example of usage:
    
    from regularizationNetworks import MixGauss
    from regularizationNetworks import separatingFKernRLS
    from regularizationNetworks import regularizedKernLSTrain
    import numpy as np
    
    lambd = 0.01
    kernel = 'gaussian'
    sigma = 1
    
    [Xtr, Ytr] = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.25'), 100)
    [Xts, Yts] = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.3'), 100)
    
    c = regularizedKernLSTrain.regularizedkernlstrain(Xtr, Ytr, 'gaussian', sigma, lambd)
    separatingFKernRLS.separatingfkernrls(c, Xtr, Ytr, 'gaussian', sigma, Xts)
    '''

    step = 0.05

    x = np.arange(xts[:, 0].min(), xts[:, 0].max(), step)
    y = np.arange(xts[:, 1].min(), xts[:, 1].max(), step)

    xv, yv = np.meshgrid(x, y)

    xv = xv.flatten('F')
    xv = np.reshape(xv, (xv.shape[0], 1))

    yv = yv.flatten('F')
    yv = np.reshape(yv, (yv.shape[0], 1))

    xgrid = np.concatenate((xv, yv), axis=1)

    ygrid = regularizedkernlstest(c, xtr, kernel, sigma, xgrid)

    '''cc = []
    for item in ytr: cc.append(colors[(int(item)+1)/2])
    plt.scatter(xtr[:, 0], xtr[:, 1], c=cc, s=50)'''
    af = plotdataset(xtr, ytr, 'separation')

    z = np.asarray(np.reshape(ygrid, (y.shape[0], x.shape[0]), 'F'))
    af.contour(x, y, z, 1)
    plt.draw()
