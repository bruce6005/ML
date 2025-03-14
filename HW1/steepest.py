import numpy as np
from LSE import matrix_inverse,generate_polynomial_features,transpose,multiply,vector_multiply

def hessian_matrix(A, m, n):
    I_m = np.identity(m)
    H = 2 * np.kron(A, I_m)
    # L, U = lu_decomposition(H)
    # b = np.ones(m * n)
    # x = solve_linear_system(L, U, b)
    print(size(H))
    return H

def steepestmethod(x_values,y_values,n):
    n=n+1
    A = generate_polynomial_features(np.array(x_values), n-1)
    b=np.asarray(y_values,dtype='float').reshape((-1,1))
    
    
    
    step_bound=100000
    step=0
    x0=np.random.rand(n,1)
   
    ATA=vector_multiply(transpose(A),A)

    AT=transpose(A)
    R=2*AT@(A@x0-b)

    alpha=0.0001
    delta=10
    while  delta > 1e-6 and step<step_bound:
        R=2*AT@(A@x0-b)
        RT=transpose(R) 
        alpha = (RT@R)/(RT@ (2*(A.T@A)) @ R)

        x1 = x0 - alpha* R


        delta=abs(np.sum(np.square(x1-x0))/n)
        x0=x1
        
        step+=1

    return x0
# import numpy as np

# def steepestmethod(x_values,y_values,n):
#     print(np.size(x_values))
#     A = generate_polynomial_features(np.array(x_values), n-1)
#     b=np.asarray(y_values,dtype='float').reshape((-1,1))
#     x=np.random.rand(n,1)
    
#     iter = 1
#     r = b - A @ x
#     delta = r.T @ r
#     conv = [delta]
#     delta0 = delta
#     maxiter=10000
#     while (delta > 1e-6 * delta0) and (iter < maxiter):
#         q = A @ r
#         alpha = delta / (r.T @ q)
#         x = x + alpha * r
        
#         if iter % 50 == 0:
#             r = b - A @ x  # Recalculate r occasionally
#         else:
#             r = r - alpha * q
        
#         delta = r.T @ r
#         conv.append(delta)
#         iter += 1

#     return x, conv

