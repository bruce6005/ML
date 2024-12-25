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
import numpy as np
def linearkernel (X):
    return X.T@X
def polynomial_kernel(X, degree=5, gamma=0.001, coef0=5):
    return (gamma * X.T@X + coef0) ** degree

def kernelPCA(X,num_dim):
    # K A = LAMBDA N A
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    X_center = standardize_data(X)
    X = standardize_data(X)
    
    kernel = polynomial_kernel(X)
    
    n_samples = kernel.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = kernel - one_n @ kernel - kernel @ one_n + one_n @ kernel @ one_n
    
    eigenvalues, eigenvectors = np.linalg.eig(K_centered)


    sort_index = np.argsort(-eigenvalues)
    if num_dim is None:
        for eigenvalue, i in zip(eigenvalues[sort_index], np.arange(len(eigenvalues))):
            if eigenvalue <= 0:
                sort_index = sort_index[:i]
                break
    else:
        sort_index=sort_index[:num_dim]

    eigenvalues=eigenvalues[sort_index]
    # from X.T@X eigenvector to X@X.T eigenvector
    eigenvectors=X_center@eigenvectors[:, sort_index]

    eigenvectors_norm=np.linalg.norm(eigenvectors,axis=0)
    eigenvectors=eigenvectors/eigenvectors_norm
    return eigenvalues,eigenvectors,X_mean
def pca(X,num_dim=None):
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    X_center = standardize_data(X)

    # PCA
    eigenvalues, eigenvectors = np.linalg.eig(X_center.T @ X_center)
    
    sort_index = np.argsort(-eigenvalues)
    if num_dim is None:
        for eigenvalue, i in zip(eigenvalues[sort_index], np.arange(len(eigenvalues))):
            if eigenvalue <= 0:
                sort_index = sort_index[:i]
                break
    else:
        sort_index=sort_index[:num_dim]

    eigenvalues=eigenvalues[sort_index]
    # from X.T@X eigenvector to X@X.T eigenvector
    eigenvectors=X_center@eigenvectors[:, sort_index]

    eigenvectors_norm=np.linalg.norm(eigenvectors,axis=0)
    eigenvectors=eigenvectors/eigenvectors_norm
    return eigenvalues,eigenvectors,X_mean


def pcam(X, num_components):
    X_standardized = standardize_data(X)

    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    # cov_matrix = covariance_matrix(X_standardized.T @ X_standardized)
    cov_matrix = X_standardized.T @ X_standardized


    eigvals, eigvecs = compute_eigenvectors(cov_matrix)
    
    
    sorted_indices = np.argsort(eigvals)[::-1]
    sorted_indices=sorted_indices[:num_components]

    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = X_standardized@eigvecs[:, sorted_indices]



    top_eigvecs = eigvecs_sorted[:, :num_components]

    # Dimension Reduction

    X_pca=top_eigvecs.T@(X_standardized)
    return X_pca,eigvals_sorted,top_eigvecs,X_mean
