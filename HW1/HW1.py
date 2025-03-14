import numpy as np
import matplotlib.pyplot as plt
import sys

from LSE import LSE_method, matrix_inverse
from newton import newtonmethod
from steepest import steepestmethod

from tool import value_predicted, calculate_error, plotdraw, line_print, param_read, file_read

def main():
    # param read
    n,lambda_val=param_read()

    # file read
    x_values,y_values = file_read()
    
    # LSE
    coef_LSE=LSE_method(n,x_values,y_values,lambda_val)

    #error compute
    predicted = value_predicted(n, coef_LSE, x_values)
    print("LSE:\nFitting Line: ", end='')
    line_print(coef_LSE,n)
    lse_error = calculate_error(y_values, predicted)
    print(f"\nLSE error: {lse_error:.20f}\n")


    ## newton
    newton_coef= newtonmethod(x_values,y_values,n)
    newton_coef= [item for sublist in newton_coef for item in sublist]

    #error compute
    newtown_predict=value_predicted(n, newton_coef, x_values) 
    newton_error = calculate_error(y_values, newtown_predict)
    #print line
    print(f"\nNewton:\nFitting Line: " ,end="")
    line_print(newton_coef,n)
    print("\nTotal error: "+str(newton_error))

    #steepest
    steepest_coef= steepestmethod(x_values,y_values,n)
    steepest_coef=steepest_coef.flatten().tolist()
    
    #error compute
    steepest_predict=value_predicted(n, steepest_coef, x_values) 
    steepest_error = calculate_error(y_values, steepest_predict)
    #print line
    print(f"\nSteepest:\nFitting Line: " ,end="")
    line_print(steepest_coef,n)
    print("\nTotal error: "+str(steepest_error))

    #plot draw
    data_points=[x_values,y_values]
    plotdraw(x_values,y_values,coef_LSE,"LSE")
    plotdraw(x_values,y_values,newton_coef,"newton")
    plotdraw(x_values,y_values,steepest_coef,"steepest")
if __name__ == "__main__":
    main()
