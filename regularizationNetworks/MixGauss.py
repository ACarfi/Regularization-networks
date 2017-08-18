import numpy as np


def mixgauss(means, sigmas, n):
    '''
    Input:
    means: (size dxp) and should be of the form [m1, ... ,mp] (each mi is d-dimensional)
    sigmas: (size px1) should be in the form [sigma_1, ... , sigma_p]
    n: number of points per class
    
    Output
    x: obtained input data matrix (size 2n x d) 
    y: obtained output data vector (size 2n)
    
    Example of usage:
    
    from regularizationNetworks import MixGauss
    import numpy as np    
    [X, Y] = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.25'), 1000)
    
    generates a 2D dataset with two classes, the first one centered on (0,0)
    with variance 0.5, the second one centered on (1,1) with variance 0.25. 
    each class will contain 1000 points
    
    to visualize:
    
    colors = ['b', 'y']
    cc = []
    for item in Y: cc.append(colors[(int(item)+1)/2])
    plt.scatter(X[:, 0], X[:, 1], c=cc, s=50)
    '''

    [d, p] = np.shape(means)

    x = np.empty([2*n, d])
    y = np.empty(2*n)

    for i in range(0, p):
        m = means[:, i]
        m = np.transpose(m)
        s = sigmas[0, i]

        x[n*i:(i+1)*n, :] = np.random.normal(m, s, [n, d])
        y[n*i:(i+1)*n] = i

    y = np.reshape(y, (y.shape[0], 1))
    y[y == 0] = -1
    return [x, y]
