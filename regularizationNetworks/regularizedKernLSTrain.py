from KernelMatrix import kernelmatrix
import numpy as np


def regularizedkernlstrain(xtr, ytr, kernel, sigma, lambd):
    '''
    Input:
    xtr: training input
    ytr: training output
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    lambd: regularization parameter
    
    Output:
    c: model weights
    
    Example of usage:
    
    from regularizationNetworks import regularizedKernLSTrain
    c =  regularizedKernLSTrain.regularizedKernLSTrain(Xtr, Ytr, 'gaussian', 1, 1e-1);
    '''
    n = xtr.shape[0]
    k = kernelmatrix(xtr, xtr, sigma, kernel)
    c = np.dot(np.linalg.pinv(k + lambd * n * np.identity(n)), ytr)

    return c
