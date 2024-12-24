import numpy as np

def standardize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def covariance_matrix(X):
    return np.cov(X.T)

def compute_eigenvectors(cov_matrix):
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    return eigvals, eigvecs

def pca(X, num_components):
    X_standardized = standardize_data(X)
    cov_matrix = covariance_matrix(X_standardized)

    eigvals, eigvecs = compute_eigenvectors(cov_matrix)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    top_eigvecs = eigvecs_sorted[:, :num_components]
    
    # Dimension Reduction
    X_pca = X_standardized.dot(top_eigvecs)
    return X_pca, eigvals_sorted, top_eigvecs
