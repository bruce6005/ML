import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2

colormap= np.random.choice(range(256),size=(100,3))
colormap =  np.array( [(255,128,0),(128,0,255),(0,128,255),(255,0,128),(50,255,128),(255,50,128),(0,255,50),(255,255,128),(255,0,255),(50,255,128)])

def imread(path):
    '''
    @param path:
    @return: (H*W,C) flatten_image ndarray
    '''
    image = cv2.imread(path)
    H, W, C = image.shape
    image_flat = np.zeros((W * H, C))
    for h in range(H):
        image_flat[h * W:(h + 1) * W] = image[h]

    return image_flat,H,W

def visualize(X,k,H,W):
    '''
    @param X: (10000) belonging classes ndarray
    @param k: #clusters
    @param H: image_H
    @param W: image_W
    @return : (H,W,3) ndarray
    '''
    colors= colormap[:k,:]
    res=np.zeros((H,W,3))
    for h in range(H):
        for w in range(W):
            res[h,w,:]=colors[X[h*W+w]]

    return res.astype(np.uint8)

def plot_eigenvector(xs,ys,zs,C,name):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    markers=['o','^','s']
    for marker,i in zip(markers,np.arange(3)):
        ax.scatter(xs[C==i],ys[C==i],zs[C==i],marker=marker)

    ax.set_xlabel('eigenvector 1st dim')
    ax.set_ylabel('eigenvector 2nd dim')
    ax.set_zlabel('eigenvector 3rd dim')
    plt.savefig(name)
    plt.show()

