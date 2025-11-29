import numpy as np
from gradient_descent import get_file, forward_propagation_2comp, compute_cost, gradient_descent_2comp
from utilis import plot_data_2comp

# Get the data
data = get_file()

# Get the time and concentration
time = np.array(data.iloc[:, 0])
time = time.reshape(1, len(time))
concentration = np.array(data.iloc[:, 1])
concentration = concentration.reshape(1, len(concentration))

# Gradient descent model
gradient_descent_parameters = gradient_descent_2comp(data)
gradient_descent_cost = compute_cost(
    forward_propagation_2comp(gradient_descent_parameters, time),
    concentration)

# Excel model
c1 = float(input("Enter your excel calculation for C1:"))
lambda1 = float(input("Enter your excel calculation for lambda1:"))
c2 = float(input("Enter your excel calculation for C2:"))
lambda2 = float(input("Enter your excel calculation for lambda2:"))

excel_parameters = {
    "c1": c1,
    "lambda1": lambda1,
    "c2": c2,
    "lambda2": lambda2
}

excel_cost = compute_cost(
    forward_propagation_2comp(excel_parameters, time),
    concentration)

print("\n\nGradient descent calculation:")
print(f"C1: {gradient_descent_parameters["c1"]}")
print(f"lambda1: {gradient_descent_parameters["lambda1"]}")
print(f"C2: {gradient_descent_parameters["c2"]}")
print(f"lambda2: {gradient_descent_parameters["lambda2"]}")
print(f"The cost with gradient descent: {gradient_descent_cost}")

print("\nExcel calculation:")
print(f"C1: {excel_parameters["c1"]}")
print(f"lambda1: {excel_parameters["lambda1"]}")
print(f"C2: {excel_parameters["c2"]}")
print(f"lambda2: {excel_parameters["lambda2"]}")
print(f"The cost with excel: {excel_cost}")

plot_data_2comp(data, gradient_descent_parameters, excel_parameters)