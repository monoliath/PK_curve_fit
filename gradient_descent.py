import numpy as np
import pandas as pd

def get_file():
    """
    Get the data from the user input
    """
    file = input("Enter the file name:")
    data = pd.read_csv(file)

    return data

def initialize_parameters_1comp(data):
    """
    Take input parameters and normalize them
    :param data: original dataframe
    :return:
    c0 -- the plasma concentration at time 0
    k -- the elimination constant
    """
    # Get input parameters.
    c0 = float(input("Enter your guess for C0:"))
    k = float(input("Enter your guess for k:"))

    # Normalize c0 and k.
    c0 = c0/np.max(data.iloc[:, 1])
    k = k*np.max(data.iloc[:, 0])

    return c0, k

def initialize_parameters_2comp(data):
    """
    Take input parameters and normalize them
    :param data: original dataframe
    :return:
    params -- a dictionary contains c1, lambda1, c2, lambda2
    """
    # Get input parameters.
    c1 = float(input("Enter your guess for C1:"))
    lambda1 = float(input("Enter your guess for lambda1:"))
    c2 = float(input("Enter your guess for C2:"))
    lambda2 = float(input("Enter your guess for lambda2:"))

    # Normalize c0 and k.
    c1 = c1/np.max(data.iloc[:, 1])
    lambda1 = lambda1*np.max(data.iloc[:, 0])
    c2 = c1/np.max(data.iloc[:, 1])
    lambda2 = lambda2*np.max(data.iloc[:, 0])

    params ={
        "c1": c1,
        "lambda1": lambda1,
        "c2": c2,
        "lambda2": lambda2
    }
    return params

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

def forward_propagation_1comp(c0, k, t):
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

def forward_propagation_2comp(params, t):
    """
    Compute c_hat
    :param params: the dictionary contains c1, lambda1, c2, lambda2
    :param t: vector contains time
    :return:
    c_hat -- the output
    """
    # Retrieve parameters
    c1 = params["c1"]
    lambda1 = params["lambda1"]
    c2 = params["c2"]
    lambda2 = params["lambda2"]

    # Compute c_hat
    c_hat = c1 * np.exp(-lambda1 * t) + c2 * np.exp(-lambda2 * t)

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

def backward_propagation_1comp(c, t, c_hat, c0, k):
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

def backward_propagation_2comp(c, t, c_hat, params):
    """
    Calculate the gradients
    :param c: vector contains real plasma concentration
    :param t: vector contains time
    :param c_hat: vector contains predicted plasma concentration
    :param params: the dictionary contains c1, lambda1, c2, lambda2
    :return:
    grads -- a dictionary contains partial derivatives with respect to c1, lambda1, c2, lambda2
    """
    # Retrieve parameters
    c1 = params["c1"]
    lambda1 = params["lambda1"]
    c2 = params["c2"]
    lambda2 = params["lambda2"]

    # Calculate the number of samples
    m = c_hat.shape[1]

    # Calculate the partial derivatives
    dL_dc1 = np.dot(np.exp(-lambda1 * t), (c_hat - c).T) * (1 / m)
    dL_dlambda1 = np.prod([np.exp(-lambda1 * t), t, (c_hat - c)], 0).sum() * (-c1/m)
    dL_dc2 = np.dot(np.exp(-lambda2 * t), (c_hat - c).T) * (1/m)
    dL_dlambda2 = np.prod([np.exp(-lambda2 * t), t, (c_hat - c)], 0).sum() * (-c2 / m)

    # Add the partial derivatives to the grads
    grads = {
        "dc1": dL_dc1,
        "dlambda1": dL_dlambda1,
        "dc2": dL_dc2,
        "dlambda2": dL_dlambda2,
    }

    return grads

def update_parameters_1comp(c0, k, grads, learn_rate=1.2):
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

def update_parameters_2comp(params, grads, learn_rate=1.2):
    """
    Update parameters based on the gradients
    :param params: the dictionary contains c1, lambda1, c2, lambda2
    :param grads: a dictionary contains partial derivatives with respect to c1, lambda1, c2, lambda2
    :learn_rate: the learn rate of gradient descent
    :return:
    updated c1, lambda1, c2, lambda2
    """
    # Retrieve parameters
    c1 = params["c1"]
    lambda1 = params["lambda1"]
    c2 = params["c2"]
    lambda2 = params["lambda2"]

    # Retrieve gradients
    dc1 = grads["dc1"]
    dlambda1 = grads["dlambda1"]
    dc2 = grads["dc2"]
    dlambda2 = grads["dlambda2"]

    # Update parameters
    c1 = c1 - learn_rate * dc1
    lambda1 = lambda1 - learn_rate * dlambda1
    c2 = c2 - learn_rate * dc2
    lambda2 = lambda2 - learn_rate * dlambda2

    params.update({
        "c1": c1,
        "lambda1": lambda1,
        "c2": c2,
        "lambda2": lambda2
    })

    return params

def gradient_descent_1comp(data, learn_rate=1.2, iteration=5000, print_cost=False):
    """
    Optimize c0 and k
    :param data: original dataframe
    :param learn_rate: the learn rate of gradient descent
    :param iteration: the number of iterations
    :param print_cost: decide to print cost value or not
    :return:
    optimized_parameters -- a dictionary contains the optimized parameters
    """
    # Get input parameters
    c0, k = initialize_parameters_1comp(data)

    # Preprocess the data
    t_norm, c_norm = preprocess(data)

    for i in range(iteration):
        # Forward propagation
        c_hat = forward_propagation_1comp(c0, k, t_norm)

        # Compute cost function
        cost = compute_cost(c_hat, c_norm)

        # Backward propagation
        grads = backward_propagation_1comp(c_norm, t_norm, c_hat, c0, k)

        # Update parameters
        c0, k = update_parameters_1comp(c0, k, grads, learn_rate)
        if print_cost and (i % 100 == 99):
            print("Cost after iteration %i: %f" % (i, cost))

    # Denormalize c0 and k
    c0 = c0 * np.max(data.iloc[:, 1])
    k = k/np.max(data.iloc[:, 0])

    # Store the result in a library
    optimized_parameters = {
        "c0": c0,
        "k": k,
    }

    return optimized_parameters

def gradient_descent_2comp(data, learn_rate=1.2, iteration=5000, print_cost=False):
    """
    Optimize c0 and k
    :param data: original dataframe
    :param learn_rate: the learn rate of gradient descent
    :param iteration: the number of iterations
    :param print_cost: decide to print cost value or not
    :return:
    optimized_parameters -- a dictionary contains the optimized parameters
    """
    # Get input parameters
    params = initialize_parameters_2comp(data)

    # Preprocess the data
    t_norm, c_norm = preprocess(data)

    for i in range(iteration):
        # Forward propagation
        c_hat = forward_propagation_2comp(params, t_norm)

        # Compute cost function
        cost = compute_cost(c_hat, c_norm)

        # Backward propagation
        grads = backward_propagation_2comp(c_norm, t_norm, c_hat, params)

        # Update parameters
        params = update_parameters_2comp(params, grads, learn_rate)
        if print_cost and (i % 100 == 99):
            print("Cost after iteration %i: %f" % (i, cost))

    # Denormalize c0 and k
    c1 = params["c1"] * np.max(data.iloc[:, 1])
    lambda1 = params["lambda1"]/np.max(data.iloc[:, 0])
    c2 = params["c2"] * np.max(data.iloc[:, 1])
    lambda2 = params["lambda2"]/np.max(data.iloc[:, 0])

    params.update({
        "c1": c1,
        "lambda1": lambda1,
        "c2": c2,
        "lambda2": lambda2
    })

    return params