# Author: Prasanga Neupane(pneupa1@lsu.edu)


import numpy as np
import matplotlib.pyplot as plt

def load_data_multivariate(data_file_path):
    input_data = np.loadtxt(data_file_path, delimiter=",")
    print("Actual Data")
    print(input_data[0:5, :])
    standarization_parameters = []
    for i in range(0, input_data.shape[1]-1):
        average = np.average(input_data[:, i])
        std = np.std(input_data[:, i])
        input_data[:, i] = (input_data[:, i]-average)/std
        standarization_parameters.append((average, std))
    print("\nData after standarization")
    print(input_data[0:5, :])

    return input_data, standarization_parameters

def calculate_mse(y, yhat):
    mse = np.sum((y-yhat)**2)/(2*y.shape[0])
    return mse

def update_weight_with_gd_multivariate(theta_one, theta_two, theta_three, theta_four, data, yhat, alpha=0.01):
    y = data[:, 3]
    x_1 = data[:, 0]
    x_2 = data[:, 1]
    x_3 = data[:, 2]

    m = data.shape[0]
    del_theta_one = (-1)*alpha*np.sum((y-yhat)*x_1)/m
    del_theta_two = (-1)*alpha*np.sum((y-yhat)*x_2)/m
    del_theta_three = (-1)*alpha*np.sum((y-yhat)*x_3)/m
    del_theta_four = (-1)*alpha*np.sum(y-yhat)/m

    theta_one = theta_one - del_theta_one
    theta_two = theta_two - del_theta_two
    theta_three = theta_three - del_theta_three
    theta_four = theta_four - del_theta_four
    return theta_one, theta_two, theta_three, theta_four

def standarize_each_value(n_beds, liv_area, lot_area, standarize_params):
    n_beds = (n_beds-standarize_params[0][0])/standarize_params[0][1]
    liv_area = (liv_area-standarize_params[1][0])/standarize_params[1][1]
    lot_area = (lot_area-standarize_params[2][0])/standarize_params[2][1]
    return n_beds, liv_area, lot_area

def main_multivariate(number_of_iterations=50, alpha=0.1, path_of_data_file=r"C:\Users\PC\Downloads\KCSmall_NS2.csv"):
    data, standarization_parms = load_data_multivariate(path_of_data_file)
    theta_one = 0
    theta_two = 0
    theta_three = 0
    theta_four = 0
    mse_list = []
    iteration_list = []
    theta_list = []
    for i in range(1, number_of_iterations+1):
        yhat = theta_one * data[:, 0] + theta_two*data[:, 1] + theta_three*data[:, 2] + theta_four
        mse_value = calculate_mse(y=data[:, 3], yhat=yhat)
        mse_list.append(mse_value)
        iteration_list.append(i)
        theta_list.append((theta_one, theta_two, theta_three, theta_three))
        if i == 0:
            print("Iteration = "+str(i)+" Loss Value = "+str(mse_value)+" theta1 = "+str(theta_one)+" theta2 = "+str(theta_two)+" theta3 = "+str(theta_three)+" theta4 = "+str(theta_four))
        theta_one, theta_two, theta_three, theta_four = update_weight_with_gd_multivariate(theta_one, theta_two, theta_three, theta_four, data, yhat, alpha=alpha)

    minimum_loss_index = mse_list.index(min(mse_list))
    minimum_loss = mse_list[minimum_loss_index]
    (learned_theta_one, learned_theta_two, learned_theta_three, learned_theta_four) = theta_list[minimum_loss_index]
    print("After " + str(number_of_iterations)+ " iterations....")
    print("Learned theta_one = "+str(theta_one)+" Learned theta_two = "+str(theta_two)+"Learned theta_three = "+str(theta_three)+" Learned theta_four = "+str(theta_four)+" Associated minimum loss = "+str(minimum_loss))
    n_bed, liv_area, lot_area = standarize_each_value(n_beds=3, liv_area=2000, lot_area=8550, standarize_params=standarization_parms)
    print("Predicted value for n_bed=3, liv_area=1180, lot_area=5650  is yhat = "+str(learned_theta_one*n_bed+learned_theta_two*liv_area+learned_theta_three*lot_area+theta_four))

    plt.plot(iteration_list, mse_list)
    plt.title("Loss Vs Iteration for Learning Rate : "+str(alpha))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

main_multivariate()