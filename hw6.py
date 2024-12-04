import cv2
from kmeans import *
import matplotlib.pyplot as plt

image = cv2.imread("./CV6/image1.png")
print(np.shape(image))
width, height, _ = image.shape

image = image.reshape(-1, 3)


# print(width, height)
# kernel_kmeans(image,5,width,height)


labels, centers = kernel_kmeans(image, 3,width,height)
print(labels)
print("")
print(centers)

centers = centers.reshape(-1, 2)
plt.scatter(image[:, 0], image[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200)
plt.title('Kernel K-Means Clustering')
plt.show()
# Print or inspect the NumPy array
print(image)

