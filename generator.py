import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import qr
from math import sqrt, log

#####################
# Signal Generation #
#####################

def generate_sparse_orthogonal_vectors(n, k, sparsity=0.5):
    """
    Generate k sparse orthogonal vectors in an n-dimensional space.
    
    Parameters:
    n (int): Dimension of the space.
    k (int): Number of orthogonal vectors (rank).
    sparsity (float): Fraction of zero entries (0 to 1).

    Returns:
    Q (ndarray): (n x k) matrix with sparse orthogonal columns.
    """
    assert 0 <= sparsity < 1, "Sparsity must be in [0,1)"
    assert k <= n, "Number of vectors must be ≤ dimension"
    
    Q = np.zeros((n, k))  # Initialize sparse matrix
    
    # Step 1: Construct sparse orthogonal vectors
    num_nonzero = max(1, int((1 - sparsity) * n))  # Number of nonzero entries per vector
    for i in range(k):
        indices = np.random.choice(n, num_nonzero, replace=False)  # Select nonzero positions
        values = np.random.randn(num_nonzero)  # Assign random values
        Q[indices, i] = values
    
    # Step 2: Orthonormalize using Gram-Schmidt
    for i in range(k):
        for j in range(i):
            Q[:, i] -= np.dot(Q[:, j], Q[:, i]) * Q[:, j]  # Remove projections
        Q[:, i] /= np.linalg.norm(Q[:, i])  # Normalize

    return Q


def generate_coherent_orthogonal_vectors(n, k, coherence=0.5):
    """
    Generate k coherent orthogonal vectors in an n-dimensional space.
    
    Parameters:
    n (int): Dimension of the space.
    k (int): Number of orthogonal vectors (rank).
    coherence (float): Coherence level (0 to 1).

    Returns:
    Q (ndarray): (n x k) matrix with coherent orthogonal columns.
    """
    assert 0 <= coherence < 1, "Coherence must be in [0,1)"
    assert k <= n, "Number of vectors must be ≤ dimension"

    mu = n ** coherence

    Q = generate_sparse_orthogonal_vectors(n, k, sparsity=mu/n)  # Generate sparse orthogonal vectors
    
    
    return Q

def generate_low_rank_sparse_signal(n, r, eigenvalues, sparsity=0.5):
    """
    Generate a symmetric low-rank matrix with sparse orthogonal eigenvectors.

    Parameters:
    n (int): Dimension of the matrix (n x n).
    r (int): Rank of the matrix.
    eigenvalues (list or array): List of r nonzero eigenvalues.
    sparsity (float): Fraction of zero entries in eigenvectors (0 to 1).

    Returns:
    A (ndarray): Symmetric low-rank matrix.
    Q (ndarray): Sparse orthogonal eigenvectors.
    """
    assert r <= n, "Rank must be ≤ n"
    assert len(eigenvalues) == r, "Eigenvalue list must match rank k"

    # Step 1: Generate sparse orthogonal vectors
    Q = generate_sparse_orthogonal_vectors(n, r, sparsity)

    # Step 2: Construct diagonal matrix with eigenvalues
    Lambda = np.diag(eigenvalues)

    # Step 3: Construct the low-rank symmetric matrix
    A = Q @ Lambda @ Q.T

    return A, Q

def generate_low_rank_coherent_signal(n, r, eigenvalues, coherence=0.5):
    """
    Generate a symmetric low-rank matrix with sparse orthogonal eigenvectors.

    Parameters:
    n (int): Dimension of the matrix (n x n).
    r (int): Rank of the matrix.
    eigenvalues (list or array): List of r nonzero eigenvalues.
    sparsity (float): Fraction of zero entries in eigenvectors (0 to 1).

    Returns:
    A (ndarray): Symmetric low-rank matrix.
    Q (ndarray): Sparse orthogonal eigenvectors.
    """
    assert r <= n, "Rank must be ≤ n"
    assert len(eigenvalues) == r, "Eigenvalue list must match rank k"

    # Step 1: Generate sparse orthogonal vectors
    Q = generate_sparse_orthogonal_vectors(n, r, coherence)

    # Step 2: Construct diagonal matrix with eigenvalues
    Lambda = np.diag(eigenvalues)

    # Step 3: Construct the low-rank symmetric matrix
    A = Q @ Lambda @ Q.T

    return A, Q


def generate_low_rank_symmetric(n, r, eigenvalues):
    """
    Generate an n x n symmetric matrix of rank r with given eigenvalues.
    
    Parameters:
    n (int): Dimension of the matrix.
    r (int): Rank of the matrix.
    eigenvalues (list or array): A list of r nonzero eigenvalues.
    
    Returns:
    np.ndarray: An n x n symmetric low-rank matrix.
    """
    assert r <= n, "Rank r must be <= matrix dimension n"
    assert len(eigenvalues) == r, "Number of eigenvalues must match rank k"

    # Step 1: Create an (n x n) random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))  # QR decomposition to ensure orthogonality

    # Step 2: Construct diagonal matrix with given eigenvalues
    Lambda = np.zeros((n, n))  # Full zero matrix
    Lambda[:r, :r] = np.diag(eigenvalues)  # Assign eigenvalues to top-left k×k block

    # Step 3: Compute the symmetric matrix
    A = Q @ Lambda @ Q.T
    return A, Q[:, :r]

















####################
# Noise Generation #
####################

def symmetric_gaussian_noise_homo(n, mean=0, var=1):
    """
    Generate an n x n symmetric matrix where all elements have the same Gaussian variance.
    
    Parameters:
    n (int): Size of the matrix (n x n).
    mean (float): Mean of the Gaussian distribution.
    var (float): Desired variance for all entries.
    
    Returns:
    np.ndarray: A symmetric matrix with Gaussian-distributed entries and uniform variance.
    """
    std = np.sqrt(var)

    # Step 1: Generate the diagonal entries
    A = np.zeros((n, n))
    np.fill_diagonal(A, np.random.normal(mean, std, size=n))

    # Step 2: Generate the upper-triangular part with variance var
    upper_triangular = np.random.normal(mean, std, size=(n, n))
    upper_triangular = np.triu(upper_triangular, k=1)  # Keep upper triangle, excluding diagonal

    # Step 3: Make the matrix symmetric
    A += upper_triangular + upper_triangular.T  # Reflect to lower triangular part

    return A

def asymmetric_gaussian_noise_homo(n, mean=0, var=1):
    A = np.random.normal(mean, sqrt(var), size=(n, n))
    return A


def symmetric_gaussian_noise_heter(n, mean=0, sdmax=1, sdmin=0.5):
    Sigma = np.zeros((n, n))
    np.fill_diagonal(Sigma, sdmin + (sdmax-sdmin) * np.random.rand(n))
    upper_triangular = sdmin + (sdmax-sdmin) * np.random.rand(n,n)
    upper_triangular = np.triu(upper_triangular, k=1)  # Keep upper triangle, excluding diagonal
    Sigma += upper_triangular + upper_triangular.T 

    # Step 3: Make the matrix symmetric
    A = symmetric_gaussian_noise_homo(n)  # Reflect to lower triangular part

    A *= Sigma
    return A



def asymmetric_gaussian_noise_heter(n, mean=0, sdmax=1, sdmin=0.5):
    Sigma = sdmin + (sdmax-sdmin) * np.random.rand(n,n)
    A = np.random.normal(size=(n, n))
    return A * Sigma
