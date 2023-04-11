import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot(x, y1, y2, y3, y4, y5, path):
    plt.plot(x, y1, label='C = 1', lw=2.0)
    plt.plot(x, y2, label='C = 2', lw=2.0)
    plt.plot(x, y3, label='C = 3', lw=2.0)
    plt.plot(x, y4, label='C = 4', lw=2.0)
    plt.plot(x, y5, label='C = 5', lw=2.0)
    plt.ylabel('Val loss', size=20, style='italic')
    plt.xlabel('Steps', size=20, style='italic')
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

steps = 800
mean_step = 100
dataset = 'kidney'
x = []
for j in range(0, int(steps/mean_step)):
    x.append(j*mean_step)
x = np.array(x)

y = []
for i in range(1, 6):
    with open('./loss_curve/' + dataset + '/' + str(i) + '/val_loss.csv', 'r') as fin:
        losses = []
        losses_mean = []
        for line in fin:
            loss = float(line.strip())
            losses.append(loss)
            if len(losses) == steps:
                break
        for j in range(int(steps/mean_step)):
            losses_mean.append(np.mean(np.array(losses)[j*mean_step: (j+1)*mean_step]))
        y.append(np.array(losses_mean))

xnew = np.linspace(x.min(),x.max(), 40)
for k in range(5):
    y[k] = make_interp_spline(x, y[k])(xnew)
plot(xnew, y[0], y[1], y[2], y[3], y[4], './loss_curve/'+ dataset +'.pdf')