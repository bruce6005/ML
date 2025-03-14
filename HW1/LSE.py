import numpy as np

# 解決線性系統 (L * y = b, U * x = y)
def solve_linear_system(L, U, b):
    n = len(L)
    y = np.zeros(n)
    x = np.zeros(n)

    # 前向替代 (L * y = b)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]

    # 反向替代 (U * x = y)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]

    return x

# LU 分解
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))

        L[i][i] = 1.0
        for k in range(i + 1, n):
            L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]

    return L, U

# 矩陣反轉
def matrix_inverse(A):
    n = len(A)
    L, U = lu_decomposition(A)
    inverse = np.zeros((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        column = solve_linear_system(L, U, e)
        for j in range(n):
            inverse[j][i] = column[j]

    return inverse

# 矩陣轉置
def transpose(matrix):
    return np.transpose(matrix)

# 矩陣乘法
def multiply(matrix1, matrix2):
    return np.dot(matrix1, matrix2)

# 向量乘法
def vector_multiply(matrix, vec):
    return np.dot(matrix, vec)

# 加上正則化
def add_regularization(matrix, lambda_):
    return matrix + lambda_ * np.identity(len(matrix))

# 生成多項式特徵
def generate_polynomial_features(x_values, degree):
    return np.vander(x_values, degree + 1, increasing=True)

# LSE 預測
def lse_predicted(n, coef, x_values):
    predict = []
    for x in x_values:
        tmp = sum(coef[j] * (x ** j) for j in range(n + 1))
        predict.append(tmp)
    return predict

# 計算 LSE 誤差
def calculate_lse(actual, predicted):
    return np.sum((np.array(actual) - np.array(predicted)) ** 2)

def LSE_method(n,x_values,y_values,Lambda):
    A = generate_polynomial_features(np.array(x_values), n)
    b = np.array(y_values)

    At = transpose(A)
    AtA = multiply(At, A)
    AtA_reg = add_regularization(AtA, Lambda)
    AtA_inverse = matrix_inverse(AtA_reg)
    Atb = vector_multiply(At, b)
    coef = vector_multiply(AtA_inverse, Atb)
    return coef