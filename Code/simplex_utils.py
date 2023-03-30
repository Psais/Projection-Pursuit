import numpy as np

def generate_rotation_matrix(d):
    """
    Generates a random d-dimensional rotation matrix.
    """
    # Generate random orthonormal matrix using QR decomposition
    H = np.random.randn(d, d)
    Q, R = np.linalg.qr(H)
    D = np.diagonal(R)
    ph = D / np.abs(D)
    Q *= ph
    
    return Q


def generate_regular_simplex(d):
    """
    Generates a d-dimensional regular simplex.
    """
    Q, R = np.linalg.qr(np.ones((d+1, d+1)))
    simplex_coords = Q[:, 1:]
    simplex_coords /= np.linalg.norm(simplex_coords[0, :] - simplex_coords[1, :])
    return Q[:, 1:]