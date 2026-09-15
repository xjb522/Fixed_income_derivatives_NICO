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

test_data = np.array([result_nm.x, result_powell.x])

print(np.abs((result_nm.x-result_powell.x)/(test_data.std())))

if np.abs((result_nm.x-result_powell.x)/(test_data.std())) < 1.96:
    print("same")
elif np.abs((result_nm.x-result_powell.x)/(test_data.std())) > 1.96:
    print("not the same")
else:
    print("What the hell")
