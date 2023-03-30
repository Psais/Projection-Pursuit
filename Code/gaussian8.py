import itertools

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pp_utils import *
import simplex_utils

# Generate random data matrix
D = 8
N = (D+1)*35
X = np.random.normal(size=(N, D))

pca = PCA(n_components=1)
pca.fit(X)

pca_fit_direction = pca.components_[0, :]

R = 0.1 * np.std(np.dot(X, pca_fit_direction))
p = 0.05
alpha = 1

def f(rs):
    return np.maximum(R-rs, np.array([0]))

# Friedman-Tukey projection index
def H(u, X):
    # Subtract the mean of X along each feature dimension
    X_centered = X - np.mean(X, axis=0)

    # Project the centered data matrix onto the u vector
    projections = np.dot(X_centered, u / np.linalg.norm(u))

    projections_trimmed = projections[np.logical_and(np.quantile(projections, p) < projections, projections < np.quantile(projections, 1-p))]

    trimmed_sd = np.std(projections_trimmed)

    projection_combis = np.fromiter(itertools.combinations(projections, 2), dtype=np.dtype((np.float64, 2)))
    local_density = np.sum(f(np.abs(projection_combis[:, 0] - projection_combis[:, 1])))

    return trimmed_sd * local_density

# Minimise this
def q(v): return -H(inverse_stereographic_projection(v), X)

def optimize_from(x0):
    res = scipy.optimize.minimize(q, x0, method="Nelder-Mead")
    if not res.success:
        raise ValueError("Optimisation failed.")
    return res.x

x0 = uniform_hypersphere(1, d=D)[0, :]
projection_direction_best = inverse_stereographic_projection(optimize_from(stereographic_projection(pca_fit_direction)))

fig = plt.figure()
ax_projected = fig.add_subplot(121)
ax_projected_pca = fig.add_subplot(122)

print("PCA objective:", H(pca_fit_direction, X))
print("PP objective:", H(projection_direction_best, X))
print("Starting point objective", H(x0, X))

projected_best = np.dot(X, projection_direction_best / np.linalg.norm(projection_direction_best))
print("Points >  4:", np.sum(projected_best >  4))
print("Points <= 4:", np.sum(projected_best <= 4))

ax_projected.hist(projected_best, bins=100)
ax_projected_pca.hist(np.dot(X, pca_fit_direction / np.linalg.norm(pca_fit_direction)), bins=100)

plt.show()
