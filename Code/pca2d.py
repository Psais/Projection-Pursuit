import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from pp_utils import *

def project(u, v, X):
    # Force u, v unit; v orthogonal to u
    u /= np.linalg.norm(u)
    v = v - np.dot(u, v) * u
    v /= np.linalg.norm(v)

    # Project the centered data matrix onto the u vector
    return X @ np.array([u, v]).T

# PCA objective function
def H(u, v, X):
    # Subtract the mean of X along each feature dimension
    X_centered = X - np.mean(X, axis=0)

    # Compute the variance of the projections
    variance = np.var(project(u, v, X_centered))

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


def q(k, l): return -H(inverse_stereographic_projection(k),
                       inverse_stereographic_projection(l), X)


def optimize_from(x0, y0, iters=200, tolerance=0.05):
    x_current = x0
    y_current = y0
    for _ in range(iters):
        res_y = scipy.optimize.minimize(lambda l: q(
            x_current, l), y0, method="Nelder-Mead")
        if not res_y.success:
            raise ValueError("Optimisation failed.")
        y_current = res_y.x
        res_x = scipy.optimize.minimize(lambda k: q(
            k, y_current), x0, method="Nelder-Mead")
        if not res_x.success:
            raise ValueError("Optimisation failed.")
        x_current = res_x.x
    if 1 - np.dot(x0 / np.linalg.norm(x0), y0 / np.linalg.norm(y0)) < tolerance:
        raise ValueError("Optimisation failed.")
    return (x0, y0)


pca = PCA(n_components=2)
pca.fit(X)

x0 = pca.components_[0, :]
y0 = pca.components_[1, :]
x0 /= np.linalg.norm(x0)
y0 /= np.linalg.norm(y0)

projection_direction1, projection_direction2 = tuple(
    inverse_stereographic_projection(vec) for vec in optimize_from(
        stereographic_projection(x0),
        stereographic_projection(y0)
    )
)


projection_direction1_points = np.array([
    projection_direction1, -projection_direction1
])

projection_direction2_points = np.array([
    projection_direction2, -projection_direction2
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
ax.plot(projection_direction1_points[:, 0], projection_direction1_points[:,
        1], projection_direction1_points[:, 2], "x--k")
ax.plot(projection_direction2_points[:, 0], projection_direction2_points[:,
        1], projection_direction2_points[:, 2], "x--k")

# Define a high resolution grid on the surface of the sphere
u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

# Evaluate the objective function on the grid
#grid_values = np.array([H(u, X) for u in grid_points])
#grid_values = grid_values.reshape(x.shape)

# Plot the heatmap of the objective function on the surface of the sphere
#ax.plot_surface(x, y, z, facecolors=plt.cm.jet(grid_values), alpha=0.3)
ax.plot_surface(x, y, z, alpha=0.1)

X_projected = project(projection_direction1, projection_direction2, X)
ax_projected.plot(X_projected[:, 0], X_projected[:, 1], "om")

plt.show()
