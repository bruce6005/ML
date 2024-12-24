import os 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def imread(path,H,W):
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
    '''
    n=int(num**0.5)
    for i in range(num):
        plt.subplot(n,n,i+1)
        plt.imshow(X[:,i].reshape(H,W),cmap='gray')
    plt.get_current_fig_manager().window.showMaximized()
    plt.show()