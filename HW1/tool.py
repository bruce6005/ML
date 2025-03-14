import numpy as np
import matplotlib.pyplot as plt
import sys

def value_predicted(n, coef, x_values):
    predictions = []
    for x in x_values:
        pred = sum(coef[j] * (x ** j) for j in range(n + 1))
        predictions.append(pred)
    return predictions

def calculate_error(actual, predicted):
    error = np.subtract(actual, predicted)
    return np.dot(error, error)

def plotdraw(x_values,y_values,coef,name):
    x_list, y_list = x_values,y_values
    plt.figure(figsize=(10, 10))
    plt.scatter(x_list, y_list, color='blue', label='Data Points')
    plt.xlim(min(x_list) - 1, max(x_list) + 1)  # X
    plt.ylim(min(y_list) - 10, max(y_list) + 10)  # Y 

    

    #curve
    coef=coef[::-1]
    poly_function = np.poly1d(coef)
    x_curve = np.linspace(min(x_list), max(x_list), 500)
    y_curve = poly_function(x_curve)
    plt.plot(x_curve, y_curve, color='red', label='Polynomial Curve')
    # 
    plt.title(name)
    plt.xlabel('X1')
    plt.ylabel('Y1')

    # 顯示圖例和網格
    plt.legend()
    plt.grid(True)

    # 顯示圖像
    plt.show()
def line_print(coef,n):
    for i in range(n, -1, -1):
        print(f"{coef[i]}", end="")
        if i != 0:
            print(f"X^{i} + ", end="")
def param_read():
    if len(sys.argv) < 3:
        print("Please input n and lambda")
        return
    n = int(sys.argv[1]) - 1
    lambda_val = float(sys.argv[2])
    return n,lambda_val

def file_read():
    with open('testfile.txt', 'r') as file:
        x_values=[]
        y_values=[]
        for line in file:
            x_str, y_str = line.strip().split(',')
            x_values.append(float(x_str))
            y_values.append(float(y_str))
    return x_values,y_values
