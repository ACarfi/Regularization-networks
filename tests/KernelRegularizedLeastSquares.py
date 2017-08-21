from regularizationNetworks.regularizedKernLSTrain import regularizedkernlstrain
from regularizationNetworks.separatingFKernRLS import separatingfkernrls
from regularizationNetworks.plotDataSet import plotdataset
from regularizationNetworks.flipLabels import fliplabels
from regularizationNetworks.two_moons import two_moons
import matplotlib.pyplot as plt
import scipy.io as sio
import os

dataset_name = 'insert_file_name'
dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets/' + dataset_name)
mat_contents = sio.loadmat(dataset_path)


Xtr = mat_contents['Xtr']
Ytr = mat_contents['Ytr']
Xts = mat_contents['Xts']
Yts = mat_contents['Yts']

plotdataset(Xtr, Ytr, 'Training Data')
plotdataset(Xts, Yts, 'Test Data')

sigma = 1
lambd = 0.1

c = regularizedkernlstrain(Xtr, Ytr, 'gaussian', sigma, lambd)
separatingfkernrls(c, Xtr, Ytr, 'gaussian', sigma, Xts)

Ytr = fliplabels(Ytr, 0.3)

c = regularizedkernlstrain(Xtr, Ytr, 'gaussian', sigma, lambd)
separatingfkernrls(c, Xtr, Ytr, 'gaussian', sigma, Xts)

Xtr, Ytr, Xts, Yts = two_moons(100, 0.05)

plotdataset(Xtr, Ytr, 'Two Moons Training Data')
plotdataset(Xts, Yts, 'Two Moons Test Data')

c = regularizedkernlstrain(Xtr, Ytr, 'gaussian', sigma, lambd)
separatingfkernrls(c, Xtr, Ytr, 'gaussian', sigma, Xts)

plt.show()
