import numpy as np
from scipy.spatial.distance import pdist, squareform,cdist
from tool import *
from PIL import Image

def kerneltwoRBF(X1, X2, length_scale=1.0, spatialalpha=1.0, coloralpha=1.0):
    #rational quadratic kernel function
    square_error=np.power(X1.reshape(-1,1)-X2.reshape(1,-1),2.0)
    
    # K = {1 + square_error /(2 * alpha * length^2)} ^ (-alpha)

    S=np.zeros((len(X1),2))
    for i in range(len(X1)):
        S[i]=[i//100,i%100]
    kernel1 = np.exp(-coloralpha * pdist(X1,'sqeuclidean'))
    print("k1 complete")
    kernel2 = np.exp(-spatialalpha * pdist(S,'sqeuclidean'))
    print("k2 complete")
    kernel  = kernel1 * kernel2
    return kernel

def save_gif(segments,gif_path):
    for i in range(len(segments)):
        segments[i] = segments[i].transpose(1, 0, 2)
    write_gif(segments, gif_path, fps=2)



def initMeans(X, k):
    # random
    Cluster = np.zeros((k, X.shape[1]))
    X_mean=np.mean(X,axis=0)
    X_std=np.std(X,axis=0)
    for c in range(X.shape[1]):
        Cluster[:,c]=np.random.normal(X_mean[c],X_std[c],size=k)
    return Cluster

def kernel_kmeans(datapoint, k_cluster, width, height):
    # datapoint 100*100
    # step1 random sample centroid
        
    Mean=initMeans(datapoint, k_cluster)
    Kernel = kerneltwoRBF(datapoint,datapoint)

    centers=np.zeros(len(datapoint),dtype=np.uint8)
    n = datapoint.shape[0]
    cnt = 0 
    diff = 1
    loss = 1e-6
    labels = np.zeros(datapoint.shape)
    segments = []

    while loss<diff:
        print(loss, diff)
        distances = np.zeros((n, k_cluster))
        for i in range(len(datapoint)):
            dist=[]
            for j in range(k_cluster):
                dist.append(np.sqrt(np.sum((datapoint[i] - Mean[j])**2)))
            centers[i]=np.argmin(dist)
        # 
        print (centers)
        labels = np.argmin(distances, axis=1)
        

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
        print(segment)
        image = Image.fromarray(segment)
        image.save(f'array_image_{cnt}.png')
        cnt+=1

    return labels, centers
