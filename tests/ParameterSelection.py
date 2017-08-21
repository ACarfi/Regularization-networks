from regularizationNetworks.regularizedKernLSTrain import regularizedkernlstrain
from regularizationNetworks.separatingFKernRLS import separatingfkernrls
from regularizationNetworks.holdoutCVKernRLS import holdoutcvkernrls
from regularizationNetworks.two_moons import two_moons
import matplotlib.pyplot as plt
import numpy as np

Xtr, Ytr, Xts, Yts = two_moons(100, 0.05)

intlambda = np.array([5, 2, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00001, 0.000001])
intkerpar = np.array([0.5])
nrip = 51
perc = 0.5

l, s, vm, vs, tm, ts = holdoutcvkernrls(Xtr, Ytr, 'gaussian', perc, nrip, intlambda, intkerpar)

f = plt.figure()
af = f.add_subplot(111)
af.set_title('median error')
#training, = af.semilogx(intlambda, tm, 'r')
#validation, = af.semilogx(intlambda, vm, 'b')
#af.legend([training, validation], ['training', 'validation'])
plt.draw()

sigma = s[0]
lambd = l[0]

c = regularizedkernlstrain(Xtr, Ytr, 'gaussian', sigma, lambd)
separatingfkernrls(c, Xtr, Ytr, 'gaussian', sigma, Xts)

plt.show()
