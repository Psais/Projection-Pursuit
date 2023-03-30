import itertools

import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pp_utils import *

df = pd.read_csv("Iris.csv")
df = df.filter(["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
X = df.to_numpy()

pca = PCA(n_components=2)
pca.fit(X)

x0 = pca.components_[0, :]
y0 = pca.components_[1, :]
x0 /= np.linalg.norm(x0)
y0 /= np.linalg.norm(y0)

def project(u, v, X):
    # Force u, v unit; v orthogonal to u
    u /= np.linalg.norm(u)
    v = v - np.dot(u, v) * u
    v /= np.linalg.norm(v)

    # Project the centered data matrix onto the u vector
    return X @ np.array([u, v]).T

R = 0.1 * np.std(np.dot(X, x0))
p = 0.05
alpha = 1

def f(rs):
    return np.maximum(R-rs, 0)

# PCA objective function
def H(u, v, X):
    # Subtract the mean of X along each feature dimension
    X_centered = X - np.mean(X, axis=0)

    projections_u = np.dot(X_centered, u)
    projections_v = np.dot(X_centered, v)

    projections_u_trimmed = projections_u[np.logical_and(np.quantile(projections_u, p) < projections_u, projections_u < np.quantile(projections_u, 1-p))]
    projections_v_trimmed = projections_v[np.logical_and(np.quantile(projections_v, p) < projections_v, projections_v < np.quantile(projections_v, 1-p))]

    trimmed_sd_u = np.std(projections_u_trimmed)
    trimmed_sd_v = np.std(projections_v_trimmed)

    projection_u_combis = np.fromiter(itertools.combinations(projections_u, 2), dtype=np.dtype((np.float64, 2)))
    projection_v_combis = np.fromiter(itertools.combinations(projections_v, 2), dtype=np.dtype((np.float64, 2)))
    local_density = np.sum(f(np.sqrt(
        np.power(np.abs(projection_u_combis[:, 0] - projection_u_combis[:, 1]), 2) +
        np.power(np.abs(projection_v_combis[:, 0] - projection_v_combis[:, 1]), 2)
    )))

    return trimmed_sd_u * trimmed_sd_v * local_density

# Minimise this
def q(k, l): return -H(inverse_stereographic_projection(k),
                       inverse_stereographic_projection(l), X)


def optimize_from(x0, y0, iters=200, tolerance=0.05):
    d = x0.shape[0]
    res = scipy.optimize.minimize(lambda kl: q(
        kl[:d], kl[d:]), np.concatenate([x0, y0]), method="BFGS")
    if not res.success:
        raise ValueError("Optimisation failed.")
    return (res.x[:d], res.x[d:])

projection_direction1, projection_direction2 = tuple(
    inverse_stereographic_projection(vec) for vec in optimize_from(
        stereographic_projection(x0),
        stereographic_projection(y0)
    )
)

# Create a 3D plot of the hypersphere and the sampled points
fig = plt.figure()
ax_projected = fig.add_subplot(121)
ax_projected_pca = fig.add_subplot(122)

X_projected = project(projection_direction1, projection_direction2, X)
X_projected_pca = project(x0, y0, X)

print("PCA objective:", H(x0, y0, X))
print("PP objective:", H(projection_direction1, projection_direction2, X))

ax_projected.plot(X_projected[:, 0], X_projected[:, 1], "om")
ax_projected_pca.plot(X_projected_pca[:, 0], X_projected_pca[:, 1], "om")

plt.show()
