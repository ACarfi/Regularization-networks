from regularizationNetworks import MixGauss
import scipy.io as sio
import numpy as np
import os.path

[Xtr, Ytr] = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.25'), 100)
[Xts, Yts] = MixGauss.mixgauss(np.matrix('0 1; 0 1'), np.matrix('0.5 0.3'), 100)

flag = True
while flag:
    file_name = raw_input('Insert the dataset name ')
    file_name = os.path.join(os.path.dirname(__file__), '..', 'datasets/'+file_name)
    if os.path.isfile(file_name + '.mat'):
        choose = raw_input('A file with this name already exists, do you want to override it? (y/n) ')
        if choose == 'y':
            flag = False
        else:
            flag = True
    else:
        flag = False
sio.savemat(file_name, {'Xtr': Xtr, 'Ytr': Ytr, 'Xts': Xts, 'Yts': Yts})

