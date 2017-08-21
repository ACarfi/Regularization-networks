import numpy as np
from regularizedKernLSTrain import regularizedkernlstrain
from regularizedKernLSTest import regularizedkernlstest


def holdoutcvkernrls(x, y, kernel, perc, nrip, intlambda, intkerpar):
    '''     
    Input:
    xtr: the training examples
    ytr: the training labels
    kernel: the kernel function (see KernelMatrix documentation).
    perc: percentage of the dataset to be used for validation
    nrip: number of repetitions of the test for each couple of parameters
    intlambda: list of regularization parameters
        for example intlambda = np.array([5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001])
    intkerpar: list of kernel parameters
        for example intkerpar = np.array([10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01])
    
    Output:
    l, s: the couple of lambda and kernel parameter that minimize the median of the validation error
    vm, vs: median and variance of the validation error for each couple of parameters
    tm, ts: median and variance of the error computed on the training set for each couple of parameters
    
    Example of usage:
    
    from regularizationNetworks import MixGauss
    from regularizationNetworks import holdoutCVKernRLS
    import matplotlib.pyplot as plt
    import numpy as np
    
    intlambda = np.array([5,2,1,0.7,0.5,0.3,0.2,0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001,0.00001,0.000001])
    intkerpar = np.array([10,7,5,4,3,2.5,2.0,1.5,1.0,0.7,0.5,0.3,0.2,0.1, 0.05, 0.03,0.02, 0.01])
    xtr, ytr = MixGauss.mixgauss([[0;0],[1;1]],[0.5,0.25],100);
    l, s, vm, vs, tm, ts = holdoutCVKernRLS.holdoutcvkernrls(xtr, ytr,'gaussian', 0.5, 5, intlambda, intkerpar);
    plt.plot(intlambda, vm, 'b')
    plt.plot(intlambda, tm, 'r')
    plt.show()
    '''

    nkerpar = intkerpar.size
    nlambda = intlambda.size

    n = x.shape[0]
    ntr = np.ceil(n*(1-perc))

    tm = np.zeros((nlambda, nkerpar))
    ts = np.zeros((nlambda, nkerpar))
    vm = np.zeros((nlambda, nkerpar))
    vs = np.zeros((nlambda, nkerpar))

    ym = float(y.max() + y.min())/float(2)

    il = 0
    for l in intlambda:
        iss = 0
        for s in intkerpar:
            trerr = np.zeros((nrip, 1))
            vlerr = np.zeros((nrip, 1))
            for rip in range(0, nrip):
                i = np.random.permutation(n)
                xtr = x[i[0:ntr], :]
                ytr = y[i[0:ntr], :]
                xvl = x[i[ntr:-1], :]
                yvl = y[i[ntr:-1], :]

                w = regularizedkernlstrain(xtr, ytr, kernel, s, l)
                trerr[rip] = calcerr(regularizedkernlstest(w, xtr, kernel, s, xtr), ytr, ym)
                vlerr[rip] = calcerr(regularizedkernlstest(w, xtr, kernel, s, xvl), yvl, ym)
                print('l: ', l, ' s: ', s, ' valErr: ', vlerr[rip], ' trErr: ', trerr[rip])
            tm[il, iss] = np.median(trerr)
            ts[il, iss] = np.std(trerr)
            vm[il, iss] = np.median(vlerr)
            vs[il, iss] = np.std(vlerr)
            iss = iss + 1
        il = il + 1
    row, col = np.where(vm == np.amin(vm))
    l = intlambda[row]
    s = intkerpar[col]

    return [l, s, vm, vs, tm, ts]


def calcerr(t, y, m):
    vt = (t >= m).astype(int)
    vy = (y >= m).astype(int)

    err = float(np.sum(vt != vy))/float(y.shape[0])
    return err