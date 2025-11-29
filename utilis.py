import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import forward_propagation_1comp, forward_propagation_2comp

def plot_data_1comp(data, c0_gd, k_gd, c0_excel, k_excel):
    """
    Plot the observed data and fitted result
    """
    # Get arrays of the observed data
    time = np.array(data.iloc[:, 0])
    concentration = np.array(data.iloc[:, 1])

    # Fit gradient descent
    c_gd = [forward_propagation_1comp(c0_gd, k_gd, i) for i in time]
    c_gd = np.array(c_gd).reshape(concentration.shape)

    # Fit excel
    c_excel = [forward_propagation_1comp(c0_excel, k_excel, i) for i in time]
    c_excel = np.array(c_excel).reshape(concentration.shape)

    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(time, concentration, label='Observed data', color='black')
    ax.plot(time, c_gd, label='Gradient Descent calculation', color='red')
    ax.plot(time, c_excel, label='Excel calculation', color='green', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('I.V. One-Compartment Model')
    plt.legend()
    plt.show()

def plot_data_2comp(data, gradient_descent_parameters, excel_parameters):
    """
    Plot the observed data and fitted result
    """
    # Retrieve parameters
    c1_gd = gradient_descent_parameters["c1"]
    lambda1_gd = gradient_descent_parameters["lambda1"]
    c2_gd = gradient_descent_parameters["c2"]
    lambda2_gd = gradient_descent_parameters["lambda2"]

    c1_excel = excel_parameters["c1"]
    lambda1_excel = excel_parameters["lambda1"]
    c2_excel = excel_parameters["c2"]
    lambda2_excel = excel_parameters["lambda2"]

    # Get arrays of the observed data
    time = np.array(data.iloc[:, 0])
    concentration = np.array(data.iloc[:, 1])

    # Fit gradient descent
    c_gd = [forward_propagation_2comp(gradient_descent_parameters, i) for i in time]
    c_gd = np.array(c_gd).reshape(concentration.shape)

    # Fit excel
    c_excel = [forward_propagation_2comp(excel_parameters, i) for i in time]
    c_excel = np.array(c_excel).reshape(concentration.shape)

    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(time, concentration, label='Observed data', color='black')
    ax.plot(time, c_gd, label='Gradient descent calculation', color='red')
    ax.plot(time, c_excel, label='Excel calculation', color='green', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('I.V. Two-Compartment Model')
    plt.legend()
    plt.show()