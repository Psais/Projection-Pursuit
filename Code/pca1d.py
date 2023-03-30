import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from pp_utils import *

# PCA objective function
def H(u, X):
    # Subtract the mean of X along each feature dimension
    X_centered = X - np.mean(X, axis=0)

    # Project the centered data matrix onto the u vector
    projections = np.dot(X_centered, u)

    # Compute the variance of the projections
    variance = np.var(projections)

    return variance

# Generate some random data matrix
X1 = np.random.normal(0, 0.1, size=(100, 3))
X2 = np.random.normal(0, 0.1, size=(100, 3))
X3 = np.random.normal(0, 0.1, size=(100, 3))
X4 = np.random.normal(0, 0.1, size=(100, 3))
mu = np.random.normal(0, 1, size=(4, 3))
X = np.concatenate((X1 + mu[0, :], X2 + mu[1, :],
                   X3 + mu[2, :], X4 + mu[3, :]), axis=0)

# Minimise this
def q(v): return -H(inverse_stereographic_projection(v), X)

def optimize_from(x0):
    res = scipy.optimize.minimize(q, x0, method="Nelder-Mead")
    if not res.success:
        raise ValueError("Optimisation failed.")
    return res.x

x0 = uniform_hypersphere(1, d=3)[0, :]

projection_direction_best = inverse_stereographic_projection(optimize_from(stereographic_projection(x0)))

projection_direction_best_points = np.array([
    projection_direction_best, -projection_direction_best
])

pca = PCA(n_components=1)
pca.fit(X)

pca_fit_direction_vec = pca.components_[0, :]
pca_fit_direction_vec /= np.linalg.norm(pca_fit_direction_vec)

pca_fit_direction = np.array([
    pca_fit_direction_vec, -pca_fit_direction_vec
])

# Create a 3D plot of the hypersphere and the sampled points
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax_projected = fig.add_subplot(122)
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

ax_projected.hist(np.dot(X, projection_direction_best / np.linalg.norm(projection_direction_best)), bins=100)

plt.show()
