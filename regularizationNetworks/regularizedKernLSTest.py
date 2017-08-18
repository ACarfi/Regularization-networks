from KernelMatrix import kernelmatrix
import numpy as np


def regularizedkernlstest(c, xtr, kernel, sigma, xts):
    '''
    Input:
    c: model weights
    xtr: training input
    kernel: type of kernel ('linear', 'polynomial', 'gaussian')
    sigma: width of the gaussian kernel, if used
    xts: test points
    
    Output:
    y: predicted model values
    
    Example of usage:
    
    from regularizationNetworks import regularizedKernLSTest
    y =  regularizedKernLSTest.regularizedkernlstest(c, Xtr, 'gaussian', 1, Xts)
    '''

    ktest = kernelmatrix(xts, xtr, sigma, kernel)
    y = np.dot(ktest, c)

    return y
