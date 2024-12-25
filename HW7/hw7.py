from PCA import *
from LDA import *
import os
from tool import *
# part 1.1  PCA LDA show first 25 eigen and fisher random 10 images and reconstruct
filepath=os.path.join('Yale_Face_Database','Training')

H,W=231,195
X,y=imread(filepath,H,W)




eigenvalues_pca,eigenvectors_pca,X_mean=kernelPCA(X,num_dim=50)
eigenvectors_pca=np.real(eigenvectors_pca)
X_pca=eigenvectors_pca.T@(X-X_mean) 
X_recover=eigenvectors_pca@X_pca+X_mean

print(np.shape(eigenvectors_pca))

show_eigenface(eigenvectors_pca,25,H,W,"DSEFkernelP.png")
show_random_ten(X,X_recover,10,H,W,"recoverkernelP.png")
X_test, y_test = imread(filepath, H, W)
acc = performance(X_test, y_test, X_pca, y, eigenvectors_pca, X_mean, 15)
print('PCA acc: {:.2f}%'.format(acc * 100))

## PCA
eigenvalues_pca,eigenvectors_pca,X_mean=pca(X,num_dim=50) 
# reduce dim (projection)
X_pca=eigenvectors_pca.T@(X-X_mean)
# X_recover= Xproj@evector.T + (X-np.mean(X, axis=0))
X_recover=eigenvectors_pca@X_pca+X_mean

show_eigenface(X,25,H,W,"EigenFace.png")   
show_eigenface(eigenvectors_pca,25,H,W,"DSEF.png")
show_random_ten(X,X_recover,10,H,W,"recover.png")
acc = performance(X_test, y_test, X_pca, y, eigenvectors_pca, X_mean, 15)
print('PCA acc: {:.2f}%'.format(acc * 100))

## LDA 
X,y=imread(filepath,H,W)
eigenvalues_pca,eigenvectors_pca,X_mean=pca(X,num_dim=35)
X_pca=eigenvectors_pca.T@(X-X_mean)
eigenvalues_lda,eigenvectors_lda=lda(X_pca,y)
U=eigenvectors_pca@eigenvectors_lda
Z=U.T@X
X_recover=U@Z+X_mean

show_eigenface(U,25,H,W,"DSEFLDA.png")
show_random_ten(X,X_recover,10,H,W,"recoverLDA.png")

filepath = os.path.join('Yale_Face_Database', 'Testing')
X_test, y_test = imread(filepath, H, W)
acc = performance(X_test, y_test, Z, y, U, X_mean, 15)
print('LDA acc: {:.2f}%'.format(acc * 100))


