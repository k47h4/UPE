import numpy as np

def phi_square(_x):
    x = np.array([_x])
    x[x<=0] = 0
    x[x>0] = x[x>0]**2 
    x[x>20] = 20
    return x[0]

def phi(x_,case=1):
    x = np.array([x_])
    if case == 1:
        x[x<=0] = 0
        x[x>20] = 20
    return x[0]

def phi_supra(_x,exponent):
    x = np.array([_x])
    x[x<=0] = 0
    x[x>0] = x[x>0]**exponent
    x[x>20] = 20
    return x[0]
