import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_file():
    """
    Get the data from the user input
    """
    file = input("Enter the file name:")
    data = pd.read_csv(file)

    return data

def initialize_parameters(data):
    """
    Take input parameters and normalize them
    :param data: original dataframe
    :return:
    c0 -- the plasma concentration at time 0
    k -- the elimination rate
    """
    # Get input parameters.
    c0 = float(input("Enter your guess for C0:"))
    k = float(input("Enter your guess for k:"))

    # Normalize c0 and k.
    c0 = c0/np.max(data.iloc[:, 1])
    k = k*np.max(data.iloc[:, 0])

    return c0, k

def preprocess(data):
    """
    Normalize, separate, and reshape the data into working vectors
    :param data: original dataframe
    :return:
    t_norm -- vector contains normalized time
    c_norm -- vector contains normalized real plasma concentration
    """
    # Separate into t and c.
    t = data.iloc[:, 0]
    c = data.iloc[:, 1]

    # Normalize data.
    t_norm = t / np.max(t)
    c_norm = c / np.max(c)

    # Reshape into vectors.
    t_norm = np.array(t_norm).reshape(1, len(t_norm))
    c_norm = np.array(c_norm).reshape(1, len(c_norm))

    return t_norm, c_norm

def forward_propagation(c0, k, t):
    """
    Compute c_hat
    :param c0: the plasma concentration at time 0
    :param k: the elimination rate
    :param t: vector contains time
    :return:
    c_hat -- the output
    """
    c_hat = c0 * np.exp(-k * t)

    return c_hat

def compute_cost(c_hat, c):
    """
    Compute the cost function
    :param c_hat: vector contains predicted plasma concentration
    :param c: vector contains real plasma concentration
    :return:
    cost -- the cost function
    """
    # Calculate the number of samples
    m = c_hat.shape[1]

    # Compute the cost function
    cost = np.sum((c_hat - c)**2)/(2*m)

    return cost

def backward_propagation(c, t, c_hat, c0, k):
    """
    Calculate the gradients
    :param c: vector contains real plasma concentration
    :param t: vector contains time
    :param c_hat: vector contains predicted plasma concentration
    :param c0: the plasma concentration at time 0
    :param k: the elimination rate
    :return:
    grads -- a dictionary contains partial derivatives with respect to c0 and k
    """
    # Calculate the number of samples
    m = c_hat.shape[1]

    # Calculate the partial derivatives
    dL_dk = np.prod([np.exp(-k * t), t, (c_hat - c)], 0).sum() * (-c0/m)
    dL_dc0 = np.dot(np.exp(-k * t), (c_hat - c).T) * (1/m)

    # Add the partial derivatives to the grads
    grads = {
        "dc0": dL_dc0,
        "dk": dL_dk,
    }

    return grads

def update_parameters(c0, k, grads, learn_rate=1.2):
    """
    Update parameters based on the gradients
    :param c0: the plasma concentration at time 0
    :param k: the elimination rate
    :param grads: a dictionary contains partial derivatives with respect to c0 and k
    :learn_rate: the learn rate of gradient descent
    :return:
    updated c0 and k
    """
    # Retrieve gradients
    dc0 = grads["dc0"]
    dk = grads["dk"]

    # Update parameters
    c0 = c0 - learn_rate * dc0
    k = k - learn_rate * dk

    return c0, k

def plot_data(data, c0, k):
    """
    Plot the observed data and fitted result
    """
    # Get arrays of the observed data
    time = np.array(data.iloc[:, 0])
    concentration = np.array(data.iloc[:, 1])

    # Fit the model
    c_fit = [forward_propagation(c0, k, i) for i in time]
    c_fit = np.array(c_fit).reshape(concentration.shape)

    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(time, concentration, label='Observed data', color='black')
    ax.plot(time, c_fit, label='Fit result', color='green')
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('I.V. One-Compartment Model: Fit result vs. Observed Data')
    plt.legend()
    plt.show()

def gradient_descent(data, learn_rate=1.2, iteration=2000, print_cost=False, plot=True):
    """
    Optimize c0 and k
    :param data: original dataframe
    :param learn_rate: the learn rate of gradient descent
    :param iteration: the number of iterations
    :param print_cost: decide to print cost value or not
    :param plot: decide to plot the data or not
    :return:
    optimized_parameters -- a dictionary contains the optimized parameters
    """
    # Get input parameters
    c0, k = initialize_parameters(data)

    # Preprocess the data
    t_norm, c_norm = preprocess(data)

    for i in range(iteration):
        # Forward propagation
        c_hat = forward_propagation(c0, k, t_norm)

        # Compute cost function
        cost = compute_cost(c_hat, c_norm)

        # Backward propagation
        grads = backward_propagation(c_norm, t_norm, c_hat, c0, k)

        # Update parameters
        c0, k = update_parameters(c0, k, grads, learn_rate)
        if print_cost:
            print("Cost after iteration %i: %f" % (i, cost))

    # Denormalize c0 and k
    c0 = c0 * np.max(data.iloc[:, 1])
    k = k/np.max(data.iloc[:, 0])

    if plot:
        plot_data(data, c0, k)

    # Store the result in a library
    optimized_parameters = {
        "c0": c0,
        "k": k,
    }

    return optimized_parameters