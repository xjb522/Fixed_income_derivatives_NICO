import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.optimize                     as sp

print("import succ")

T = np.linspace(0.5,10,20)

Rate = (0.06/2)*100

C = np.ones(20) * Rate

C[19] = 3 + 100

def pv(y, T, C):
    PI =np.sum(C / (1 + y)**T)

    return PI

price = 98.74

def objective(y, price, T, C):

    if y <= -1:
        return 1e20

    model_price = pv(y, T, C)
    SE = (price - model_price)**2

    return SE


result_nm = sp.minimize(
    objective,
    x0=0.05,
    args=(price, T, C),
    method="Nelder-Mead"
)

result_powell = sp.minimize(
    objective,
    x0=0.05,
    args=(price, T, C),
    method="Powell"
)

print(result_nm.x)
print(result_powell.x)



def FFF(x):
    y = x**2 + 2*x - 5
    return y

maxXx = sp.minimize(
    FFF,
    x0=12,
    method='Nelder-Mead'
)

print(maxXx.x)


YYY = np.random.normal(10,100,100)
XXX = np.random.uniform(0.20,100,100)

X = np.column_stack([
    np.ones(100),
    XXX
])

Y = YYY

print(X.shape)
print(Y.shape)

# OLS: beta_hat = (X'X)^(-1) X'Y
XTX = X.T @ X
XTY = X.T @ Y

beta_hat = np.linalg.inv(XTX) @ XTY

print(beta_hat)
