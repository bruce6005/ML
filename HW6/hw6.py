import cv2
from kmeans import *
import matplotlib.pyplot as plt
import time
import os


imgname="img2"
image = cv2.imread("./CV6/image2.png")

width, height, _ = image.shape

image = image.reshape(-1, 3)


start_time = time.time()
K = kernel(image,image)
D = np.diag(K.sum(axis=1))  # Degree matrix
L_Normal = D - K
D_inv=np.diag(1/np.diag(np.sqrt(D)))
L_Ratio=D_inv@L_Normal@D_inv

eigvals_Normal, eigvecs_Noraml = np.linalg.eigh(L_Normal)
eigvals_Ratio, eigvecs_Ratio = np.linalg.eigh(L_Ratio)
end_time = time.time()
print(f"Execution time: {end_time - start_time:.6f} seconds")
# save_dir = "output2/Kmeans_" + imgname + "_Cluster" + str(2) + "_initMode_" + str(2)
# kernel_kmeans(image, 2, width,height, save_dir, 2)

clusterNum = 10
init =1
save_dir = "output4/SC_Normal_" + imgname + "_Cluster" + str(clusterNum) + "_initMode_" + str(init)
spectral_clustering( clusterNum, width,height, save_dir, init, eigvals_Normal, eigvecs_Noraml)

# Ratio spectral clustering
save_dir = "output4/SC_Ratio_" + imgname+"_Cluster" + str(clusterNum) + "_initMode_" + str(init)
spectral_clustering( clusterNum, width,height, save_dir, init, eigvals_Ratio, eigvecs_Ratio)

# kernel kmeans
save_dir = "output4/Kmeans_" + imgname + "_Cluster" + str(clusterNum) + "_initMode_" + str(init)
kernel_kmeans(image, clusterNum, width,height, save_dir, init)

# 2 - 4 cluster try
# for clusterNum in range(2,5):
#     # mode kmeans++ : 1  random : 2 
#     for init in range (1,3):
#         # normal spectral clustering
#         save_dir = "output3/SC_Normal_" + imgname + "_Cluster" + str(clusterNum) + "_initMode_" + str(init)
#         spectral_clustering( clusterNum, width,height, save_dir, init, eigvals_Normal, eigvecs_Noraml)

#         # Ratio spectral clustering
#         save_dir = "output3/SC_Ratio_" + imgname+"_Cluster" + str(clusterNum) + "_initMode_" + str(init)
#         spectral_clustering( clusterNum, width,height, save_dir, init, eigvals_Ratio, eigvecs_Ratio)

#         # kernel kmeans
#         # save_dir = "output2/Kmeans_" + imgname + "_Cluster" + str(clusterNum) + "_initMode_" + str(init)
#         # kernel_kmeans(image, clusterNum, width,height, save_dir, init)


# print("\a")


