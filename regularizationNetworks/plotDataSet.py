import matplotlib.pyplot as plt


def plotdataset(x, y, name):
    '''
    Input:
    
    x: 2D points coordinate
    y: label of each point, only two labels are accepted
    name: a string containing the title of the plot
    '''
    colors = ['b', 'y']
    cc = []
    for item in y: cc.append(colors[(int(item) + 1) / 2])
    f = plt.figure()
    af = f.add_subplot(111)
    af.set_title(name)
    af.scatter(x[:, 0], x[:, 1], c=cc, s=50)
    plt.draw()

    return af
