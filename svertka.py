import numpy as np



x = [2,2,3,4,5]

h = [2,2,3]

def svertka(x,h):



    res = len(x) +len(h) - 1 #длина выходного вектора


    y = np.zeros(res) # выходной вектор

    for k in range(res):
        for m in range(len(h)):
            if 0 <= k-m < len(x):
                y[k]+=h[m]*x[k-m]


    return y



def cp_svertka(x,h):

    cp_len= len(h) - 1#длина префикса

    x_cp = np.concatenate((x[-cp_len: ], x)) # с префиксом х

    print(x_cp)

    res = len(x_cp) +len(h) - 1

    y = np.zeros(res)

    for k in range(res):
        for m in range(len(h)):
            if 0 <= k - m < len(x_cp):
                y[k] += h[m] * x_cp[k - m]
    print(y)

    return y[cp_len:-cp_len]






print(svertka(x,h))
print(cp_svertka(x,h))