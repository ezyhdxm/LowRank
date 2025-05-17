import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import qr
from math import sqrt, log


def top_r_low_rank_symmetric(A, r):
    """
    Compute the best rank-r approximation of a symmetric matrix A using eigendecomposition.
    
    Parameters:
    A (ndarray): A symmetric (n x n) matrix.
    r (int): The desired rank of the approximation (must be ≤ n).
    
    Returns:
    A_r (ndarray): The best rank-r approximation of A.
    eigvals_r (ndarray): The top r eigenvalues.
    eigvecs_r (ndarray): The corresponding eigenvectors.
    """
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    assert r > 0 and r <= A.shape[0], "Rank r must be between 1 and n"

    # Step 1: Compute eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(A)  # eigh is optimized for symmetric matrices

    # Step 2: Sort eigenvalues in descending order
    idx = np.argsort(-np.abs(eigvals))  # Get indices to sort in descending order
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 3: Keep only the top-r components
    eigvals_r = eigvals[:r]  # Select top-r eigenvalues
    eigvecs_r = eigvecs[:, :r]  # Select corresponding eigenvectors

    # Step 4: Reconstruct the low-rank matrix
    A_r = eigvecs_r @ np.diag(eigvals_r) @ eigvecs_r.T

    return A_r, eigvecs_r, eigvals_r

def low_rank_entrywise(A, r):
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    assert r > 0 and r <= A.shape[0], "Rank r must be between 1 and n"

    # Step 1: Compute eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(A)  # eigh is optimized for symmetric matrices

    # Step 2: Sort eigenvalues in descending order
    idx = np.argsort(-np.abs(eigvals))  # Get indices to sort in descending order
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Step 3: Keep only the top-r components
    eigvals_r = eigvals[:r]  # Select top-r eigenvalues
    eigvecs_r = eigvecs[:, :r]  # Select corresponding eigenvectors

    # Step 4: Reconstruct the low-rank matrix
    A_r = eigvecs_r @ np.diag(eigvals_r) @ eigvecs_r.T

    return A_r


    

def top_r_low_rank_asymmetric(A, r):
    """
    Compute the best rank-r approximation of a symmetric matrix A using eigendecomposition.
    
    Parameters:
    A (ndarray): A symmetric (n x n) matrix.
    r (int): The desired rank of the approximation (must be ≤ n).
    
    Returns:
    A_r (ndarray): The best rank-r approximation of A.
    eigvals_r (ndarray): The top r eigenvalues.
    eigvecs_r (ndarray): The corresponding eigenvectors.
    """
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    assert r > 0 and r <= A.shape[0], "Rank r must be between 1 and n"

    # Step 1: Compute eigendecomposition
    Reigvals, Reigvecs = np.linalg.eig(A) 
    Leigvals, Leigvecs = np.linalg.eig(A.T) 

    # Step 2: Sort eigenvalues in descending order
    idx = np.argsort(-np.abs(Reigvals))  # Get indices to sort in descending order
    Reigvals_r = Reigvals[idx[:r]]
    Reigvecs_r = Reigvecs[:, idx[:r]]

    # Step 3: Keep only the top-r components
    idx = np.argsort(-np.abs(Leigvals))  # Get indices to sort in descending order
    Leigvals_r = Leigvals[idx[:r]]  # Select top-r eigenvalues
    Leigvecs_r = Leigvecs[:, idx[:r]]  # Select corresponding eigenvectors

    eigvals_r = Reigvals_r

    return Leigvecs_r, Reigvecs_r, eigvals_r

def top_r_low_rank_asymmetric_svd(A, r):
    """
    Compute the best rank-r approximation of a symmetric matrix A using eigendecomposition.
    
    Parameters:
    A (ndarray): A symmetric (n x n) matrix.
    r (int): The desired rank of the approximation (must be ≤ n).
    
    Returns:
    A_r (ndarray): The best rank-r approximation of A.
    eigvals_r (ndarray): The top r eigenvalues.
    eigvecs_r (ndarray): The corresponding eigenvectors.
    """
    assert A.shape[0] == A.shape[1], "Matrix must be square"
    assert r > 0 and r <= A.shape[0], "Rank r must be between 1 and n"

    # Step 1: Compute eigendecomposition
    U, S, Vh = np.linalg.svd(A) 
    
    # Step 2: Sort eigenvalues in descending order
    idx = np.argsort(-np.abs(S))  # Get indices to sort in descending order
    S_r = S[idx[:r]]
    U_r = U[:, idx[:r]]
    Vh_r = Vh[idx[:r], :]

    return U_r, S_r, Vh_r


