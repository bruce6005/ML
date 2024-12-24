import numpy as np
from scipy.spatial.distance import pdist, squareform,cdist
from tool import *
from PIL import Image
import os

def kernel(X1, X2, spatialalpha=0.001, coloralpha=0.01):

    S=np.zeros((len(X1),2))
    for i in range(len(X1)):
        S[i]=[i//100,i%100]

    K=squareform(np.exp(-spatialalpha*pdist(S,'sqeuclidean')))*squareform(np.exp(-coloralpha*pdist(X1,'sqeuclidean')))
    return K

def initMeans(X, k, mode):
    Cluster = np.zeros((k, X.shape[1]))

    # k-means++
    if mode ==1:
        Cluster[0]=X[np.random.randint(low=0,high=X.shape[0],size=1),:]
        for c in range(1,k):
            Dist=np.zeros((len(X),c))
            for i in range(len(X)):
                for j in range(c):
                    Dist[i,j]=np.sqrt(np.sum((X[i]-Cluster[j])**2))
            Dist_min=np.min(Dist,axis=1)
            sum=np.sum(Dist_min)*np.random.rand()
            for i in range(len(X)):
                sum-=Dist_min[i]
                if sum<=0:
                    Cluster[c]=X[i]
    # random
    elif mode ==2:
        X_mean=np.mean(X,axis=0)
        X_std=np.std(X,axis=0)
        for c in range(X.shape[1]):
            Cluster[:,c]=np.random.normal(X_mean[c],X_std[c],size=k)
    
    
    
    return Cluster
    
def kernel_kmeans(datapoint, k_cluster, width, height,dir,initmode):
    datapoint = kernel(datapoint,datapoint)
    kmeans(datapoint, k_cluster, width, height,dir,initmode)

def kmeans(datapoint, k_cluster, width, height, dir, initmode):
    # datapoint 100*100
    # step1 random sample centroid
    
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directory '{dir}' created.")
    else:
        print(f"Directory '{dir}' already exists.")

    Mean=initMeans(datapoint, k_cluster,initmode)
    

    centers=np.zeros(len(datapoint),dtype=np.uint8)
    n = datapoint.shape[0]
    cnt = 0 
    diff = 1
    loss = 1e-6
    labels = np.zeros(datapoint.shape)
    segments = []

    while loss<diff :
        distances = np.zeros((n, k_cluster))
        for i in range(len(datapoint)):
            dist=[]
            for j in range(k_cluster):
                dist.append(np.sqrt(np.sum((datapoint[i] - Mean[j])**2)))
            centers[i]=np.argmin(dist)
        # 
        # labels = np.argmin(distances, axis=1)
        

        New_Mean=np.zeros(Mean.shape)
        # recalculate centroid
        for j in range(k_cluster):
            belong=np.argwhere(centers==j).reshape(-1)
            for k in belong:
                New_Mean[j]=New_Mean[j]+datapoint[k]
            if len(belong)>0:
                New_Mean[j]=New_Mean[j]/len(belong)


            # cluster_points = datapoint[labels == j]
            # cluster_kernel_matrix = Kernel[labels == j, :]
            # centers[j] = np.argmin(np.sum(cluster_kernel_matrix, axis=0))  #
            
        diff =np.sum((New_Mean - Mean)**2)
        Mean = New_Mean
        


        segment = visualize(centers,k_cluster,height,width)
        segments.append(segment)
        image = Image.fromarray(segment)
        
        
        save_path = os.path.join(dir, f'array_image_{cnt}.png')
        image.save(save_path)
        
        print(cnt,loss, diff)
        cnt+=1
    # print(centers)
    return centers, centers

def spectral_clustering(k_cluster, width, height, dir, initmode, eigvals,eigvecs):
    # eigencalue count before this function
    sort_index=np.argsort(eigvals)
    # k-means
    U=eigvecs[:,sort_index[1:1+k_cluster]]
    sums=np.sqrt(np.sum(np.square(U),axis=1)).reshape(-1,1)
    T=U/sums
    labels, _ = kmeans(T,k_cluster,width, height,dir,initmode)
    
    
    if k_cluster==3:
        plot_eigenvector(U[:,0],U[:,1],U[:,2],labels,dir+".jpg")
    