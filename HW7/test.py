import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def imread(path,H,W):
    '''
    :param path:
    :param H:
    :param W:
    :return:  (W*H, # pics) ndarray , (# pics) ndarray
    '''
    pics=os.listdir(path)
    images=np.zeros((W*H,len(pics)))
    labels=np.zeros(len(pics)).astype('uint8')
    for pic,i in zip(pics,np.arange(len(pics))):
        labels[i]=int(pic.split('.')[0][7:9])-1
        image=np.asarray(Image.open(os.path.join(path,pic)).resize((W,H),Image.Resampling.LANCZOS)).flatten()
        images[:,i]=image

    return images,labels

def show_eigenface(X,num,H,W):
    '''
    :param X: (H*W, low-dim) ndarray
    :param num: # of showing faces
    :param H:
    :param W:
    :return: 
    '''
    n=int(num**0.5)
    for i in range(num):
        plt.subplot(n,n,i+1)
        plt.imshow(X[:,i].reshape(H,W),cmap='gray')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

def show_reconstruction(X,X_recover,num,H,W):
    '''
    :param X:  (H*W,# person picture=135) ndarray
    :param X_recover:  (H*W# person picture=135) ndarray
    :param num:  # of showing faces
    :param H:
    :param W:
    :return:
    '''
    randint=np.random.choice(X.shape[1],num)
    for i in range(num):
        plt.subplot(2,num,i+1)
        plt.imshow(X[:,randint[i]].reshape(H,W),cmap='gray')
        plt.subplot(2,num,i+1+num)
        plt.imshow(X_recover[:,randint[i]].reshape(H,W),cmap='gray')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()

def performance(X_test,y_test,Z_train,y_train,U,X_mean=None,k=3):
    '''
    using k-nn to predict X_test's label
    :param X_test:  (H*W, # pics) ndarray
    :param y_test:   (# pics) ndarray
    :param Z_train:  (low-dim, #pics) ndarray
    :param y_train:  (# pics) ndarray
    :param U: Transform matrix
    :param X_mean:  using when estimate eigenface
    :param k: k of k-nn
    :return:
    '''
    if X_mean is None:
        X_mean=np.zeros((X_test.shape[0],1))

    # reduce dim (projection)
    Z_test=U.T@(X_test-X_mean)

    # k-nn
    predicted_y=np.zeros(Z_test.shape[1])
    for i in range(Z_test.shape[1]):
        distance=np.zeros(Z_train.shape[1])
        for j in range(Z_train.shape[1]):
            distance[j]=np.sum(np.square(Z_test[:,i]-Z_train[:,j]))
        sort_index=np.argsort(distance)
        nearest_neighbors=y_train[sort_index[:k]]
        unique, counts = np.unique(nearest_neighbors, return_counts=True)
        nearest_neighbors=[k for k,v in sorted(dict(zip(unique, counts)).items(), key=lambda item: -item[1])]
        predicted_y[i]=nearest_neighbors[0]

    acc=np.count_nonzero((y_test-predicted_y)==0)/len(y_test)
    return acc
def lda(X,y,num_dim=None):
    N=X.shape[0]
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    
    classes_mean = np.zeros((N, 15))  # 15 classes
    for i in range(X.shape[1]):
        classes_mean[:, y[i]] += X[:, y[i]]
    classes_mean = classes_mean / 9

    # within-class scatter
    S_within = np.zeros((N, N))
    print(np.shape(X))
    for i in range(X.shape[1]):
        d = X[:, y[i]].reshape(-1,1) - classes_mean[:, y[i]].reshape(-1,1)
        S_within += d @ d.T

    # between-class scatter
    S_between = np.zeros((N, N))
    for i in range(15):
        d = classes_mean[:, i].reshape(-1,1) - X_mean
        S_between += 9 * d @ d.T

    eigenvalues,eigenvectors=np.linalg.eig(np.linalg.inv(S_within)@S_between)
    sort_index=np.argsort(-eigenvalues)
    if num_dim is None:
        sort_index=sort_index[:-1]  # reduce 1 dim
    else:
        sort_index=sort_index[:num_dim]

    eigenvalues=np.asarray(eigenvalues[sort_index].real,dtype='float')
    eigenvectors=np.asarray(eigenvectors[:,sort_index].real,dtype='float')

    return eigenvalues,eigenvectors

import numpy as np


def pca(X,num_dim=None):
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    X_center = X - X_mean

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
    print("123")
    print(np.shape(eigenvectors))

    eigenvectors_norm=np.linalg.norm(eigenvectors,axis=0)
    eigenvectors=eigenvectors/eigenvectors_norm
    print("PCA")
    print(np.shape(eigenvectors))
    print(np.shape(eigenvalues))
    return eigenvalues,eigenvectors,X_mean

import numpy as np
import os
import matplotlib.pyplot as plt

if __name__=='__main__':
    filepath=os.path.join('Yale_Face_Database','Training')
    H,W=231,195
    X,y=imread(filepath,H,W)

    eigenvalues_pca,eigenvectors_pca,X_mean=pca(X,num_dim=31)
    X_pca=eigenvectors_pca.T@(X-X_mean)
    eigenvalues_lda,eigenvectors_lda=lda(X_pca,y)

    # Transform matrix
    U=eigenvectors_pca@eigenvectors_lda
    print(np.shape(eigenvectors_pca),np.shape(eigenvectors_lda))
    print('U shape: {}'.format(U.shape))

    # show top 25 eigenface
    # show_eigenface(U,25,H,W)

    # reduce dim (projection)
    Z=U.T@X

    # recover
    X_recover=U@Z+X_mean
    # show_reconstruction(X,X_recover,10,H,W)

    # accuracy
    filepath = os.path.join('Yale_Face_Database', 'Testing')
    X_test, y_test = imread(filepath, H, W)
    acc = performance(X_test, y_test, Z, y, U, X_mean, 5)
    print('acc: {:.2f}%'.format(acc * 100))