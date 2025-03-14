import numpy as np
from LSE import matrix_inverse,generate_polynomial_features,transpose,multiply,vector_multiply

# from linearalgo.linalg import inverse

def newtonmethod(x_values,y_values,n):
    n=n+1
    A = generate_polynomial_features(np.array(x_values), n-1)
    b=np.asarray(y_values,dtype='float').reshape((-1,1))
    
    step_bound=10000
    step=0
    x0=np.random.rand(n,1)
    eps=100
    while eps>1e-6 and step<step_bound:
        
        ATA = multiply(transpose(A), A)
        ATA = multiply(ATA, np.identity(n) * 2)
        ATb = vector_multiply(transpose(A), b)
        ATb = vector_multiply( np.identity(n) * 2,ATb)
        inv_ATA = matrix_inverse(ATA)
        term = vector_multiply(ATA, x0)
        term = vector_multiply(inv_ATA, term - ATb)
        x1 = x0 - term
        
        step+=1
        eps=abs(np.sum(np.square(x1-x0))/n)
        x0=x1
    return x0
