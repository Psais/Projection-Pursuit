import numpy as np

def uniform_hypersphere(n, d):
    # Generate n uniformly distributed points on a d-dimensional hypersphere
    x = np.random.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x

def stereographic_projection(u):
    return u[1:]/(1-u[0])

def inverse_stereographic_projection(u):
    s2 = np.sum(u**2)
    return np.concatenate([[(s2-1)/(s2+1)], 2*u/(s2+1)])