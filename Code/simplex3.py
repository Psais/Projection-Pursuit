import itertools

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from pp_utils import *
import simplex_utils

# Generate some random data matrix
X1 = np.random.normal(0, 0.1, size=(30, 3))
X2 = np.random.normal(0, 0.1, size=(30, 3))
X3 = np.random.normal(0, 0.1, size=(30, 3))
X4 = np.random.normal(0, 0.1, size=(30, 3))
mu = simplex_utils.generate_regular_simplex(3) @ simplex_utils.generate_rotation_matrix(3)
X = 2 * np.concatenate((X1 + mu[0, :], X2 + mu[1, :],
                   X3 + mu[2, :], X4 + mu[3, :]), axis=0)
X = X - np.mean(X, axis=0)

pca = PCA(n_components=1)
pca.fit(X)

pca_fit_direction_vec = pca.components_[0, :]
pca_fit_direction_vec /= np.linalg.norm(pca_fit_direction_vec)

pca_fit_direction = np.array([
    pca_fit_direction_vec, -pca_fit_direction_vec
])

R = 0.1 * np.std(np.dot(X, pca_fit_direction_vec))
p = 0.05
alpha = 1

def f(rs):
    return np.maximum(R-rs, 0)

# PCA objective function
def H(u, X):
    # Subtract the mean of X along each feature dimension
    X_centered = X - np.mean(X, axis=0)

    # Project the centered data matrix onto the u vector
    projections = np.dot(X_centered, u)

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

projection_direction_best = inverse_stereographic_projection(optimize_from(stereographic_projection(pca_fit_direction_vec)))

projection_direction_best_points = np.array([
    projection_direction_best, -projection_direction_best
])

# Create a 3D plot of the hypersphere and the sampled points
fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax_projected = fig.add_subplot(132)
ax_projected_pca = fig.add_subplot(133)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sampling on Hypersphere')

ax.plot(X[:, 0], X[:, 1], X[:, 2], "om")
ax.plot(projection_direction_best_points[:, 0], projection_direction_best_points[:, 1], projection_direction_best_points[:, 2], "o-c")
ax.plot(pca_fit_direction[:, 0], pca_fit_direction[:, 1], pca_fit_direction[:, 2], "x--k")

# Define a high resolution grid on the surface of the sphere
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

# Evaluate the objective function on the grid
grid_values = np.array([H(u, X) for u in grid_points])
grid_values = grid_values.reshape(x.shape)

# Plot the heatmap of the objective function on the surface of the sphere
ax.plot_surface(x, y, z, facecolors=plt.cm.jet(grid_values), alpha=0.3)

print("PCA objective:", H(pca_fit_direction_vec, X))
print("PP objective:", H(projection_direction_best, X))

ax_projected.hist(np.dot(X, projection_direction_best / np.linalg.norm(projection_direction_best)), bins=100)
ax_projected_pca.hist(np.dot(X, pca_fit_direction_vec / np.linalg.norm(pca_fit_direction_vec)), bins=100)

plt.show()
