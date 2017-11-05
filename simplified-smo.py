import numpy as np
import scipy.io
import random
import matplotlib.pyplot as plt

def smo(c, tol, max_passes, x_trn, y_trn):
    passes = 0
    alphas = []
    alphas_old = []
    for i in range(y_trn.size):
        alphas.append(0)
        alphas_old.append(0)
    b = 0
    while passes < max_passes:
        num_changes_alphas = 0
        for i in range(y_trn.size):
            err_i = error_i(x_trn[i, :], y_trn[i, 0], alphas[i], b)
            if (y_trn[i, 0]*err_i < -tol and alphas[i] < c) or (y_trn[i, 0]*err_i > tol and alphas[i] > 0):
                j = 0
                while j == i:
                    j = random.randint(1, y_trn.size)
                err_j = error_i(x_trn[j, :], y_trn[j, 0], alphas[j], b)
                alphas_old[i] = alphas[i]
                alphas_old[j] = alphas[j]
                if y_trn[i, 0] != y_trn[j, 0]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c+alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[i] + alphas[j] - c)
                    h = min(c, alphas[i] + alphas[j])
                if l == h:
                    continue
                neta = 2*np.dot(x_trn[i, :], x_trn[j, :].T) - np.dot(x_trn[i, :], x_trn[i, :].T) - np.dot(x_trn[j, :], x_trn[j, :].T)
                neta = neta[0, 0]
                if neta >= 0:
                    continue
                alphas[j] = alphas[j] - y_trn[j, 0]*(err_i - err_j)/neta
                if alphas[j] > h:
                    alphas[j] = h
                elif alphas[j] < l:
                    alphas[j] = l
                if abs(alphas[j] - alphas_old[j]) < 0.00001:
                    continue
                alphas[i] += y_trn[i, 0]*y_trn[j, 0]*(alphas_old[j] - alphas[j])
                b1 = b - err_i - y_trn[i, 0]*(alphas[i] - alphas_old[i])*np.dot(x_trn[i, :], x_trn[i, :].T) - y_trn[j, 0]*(alphas[j] - alphas_old[j])*np.dot(x_trn[i, :], x_trn[j, :].T)
                b2 = b - err_j - y_trn[i, 0]*(alphas[i] - alphas_old[i])*np.dot(x_trn[i, :], x_trn[i, :].T) - y_trn[j, 0]*(alphas[j] - alphas_old[j])*np.dot(x_trn[i, :], x_trn[j, :].T)
                if alphas[i] < c and alphas[i] > 0:
                    b = b1
                elif alphas[j] < c and alphas[j] > 0:
                    b = b2
                else:
                    b = (b1 + b2)/2
                num_changes_alphas += 1
        if num_changes_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alphas, b


def error_i(x, y, alpha, b):
    return (alpha*y*np.dot(x, x.T) + b - y)[0, 0]


c = 0.1
tol = 0.1
max_passes = 2
data = scipy.io.loadmat('data2.mat')
x_trn = np.asmatrix(data['X_trn'])
y_trn = np.asmatrix(data['Y_trn'])
for i in range(y_trn.size):
    if y_trn[i, 0] == 0:
        y_trn[i, 0] = -1
x_tst = np.asmatrix(data['X_tst'])
y_tst = np.asmatrix(data['Y_tst'])


alpha, b_final = smo(c, tol, max_passes, x_trn, y_trn)
w_opt = 0
for i in range(y_trn.size):
    w_opt += alpha[i]*y_trn[i, 0]*x_trn[i, :]

prediction_trn = np.matmul(x_trn, w_opt.T)
prob_trn = np.zeros(y_trn.shape)
x2 = np.zeros((x_trn.shape[0], 1))
for i in range(y_trn.size):
    x2[i, 0] = -1*(w_opt[0, 0]*x_trn[i, 0])/w_opt[0, 1]
for i in range(y_trn.size):
    if y_trn[i, 0] == 1:
        plt.scatter([x_trn[i, 0]], [x_trn[i, 1]], c='red')
    else:
        plt.scatter([x_trn[i, 0]], [x_trn[i, 1]], c='blue')
plt.plot(x_trn[:, 0], x2[:, 0], color='black')
plt.show()


print(w_opt)
