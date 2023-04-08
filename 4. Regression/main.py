from pandas import Series, DataFrame
import pandas as pd
from sklearn.linear_model import LinearRegression

def mean(arr):
    return sum(arr)/len(arr)

def lineur_regression(x_arr, y_arr):
    if (len(x_arr) != len(y_arr)):
        raise Exception("ERROR: length not equals")
    x_mean = mean(x_arr)
    y_mean = mean(y_arr)
    num_sum, den_sum = 0, 0
    for i in range(len(x_arr)):
        num_sum += (x_arr[i] - x_mean) * (y_arr[i] - y_mean)
        den_sum += (x_arr[i] - x_mean) ** 2
    theta_1 = num_sum / den_sum
    theta_0 = y_mean - theta_1 * x_mean

    # caclulate R
    num_sum, den_sum = 0, 0
    for i in range(len(x_arr)):
        num_sum += (y_arr[i] - theta_0 - theta_1 * x_arr[i]) ** 2
        den_sum += (y_arr[i] - y_mean) ** 2
    r_pow_2 = 1 - num_sum / den_sum

    print("theta_1 = ", theta_1)
    print("theta_0 = ", theta_0)
    print("r_pow_2 = ", r_pow_2)



def main():
    # task-1
    x_arr = [6, 14, 10, 16, 2, 1, 21, 7, 25, 12]
    y_arr = [12, 23, 32, 44, 5, 5, 40, 15, 40, 34]
    x_mean = mean(x_arr)
    y_mean = mean(y_arr)
    print("Выборочное среднее X: ", x_mean)
    print("Выборочное среднее Y: ", y_mean)
    lineur_regression(x_arr, y_arr)

    # task-2



if __name__ == "__main__":
    main()