import numpy as np
from gradient_descent import get_file, forward_propagation, compute_cost, gradient_descent

# Get the data
data = get_file()

# Get the time and concentration
time = np.array(data.iloc[:, 0])
time = time.reshape(1, len(time))
concentration = np.array(data.iloc[:, 1])
concentration = concentration.reshape(1, len(concentration))

# Gradient descent model
gradient_descent_parameters = gradient_descent(data, plot=True)
gradient_descent_cost = compute_cost(
    forward_propagation(gradient_descent_parameters["c0"],
    gradient_descent_parameters["k"], time),
    concentration)

# Excel model
k_excel = float(input("Enter your excel calculation of the elimination rate:"))
c0_excel = float(input("Enter your excel calculation of the initial plasma concentration:"))
excel_cost = compute_cost(
    forward_propagation(k_excel, c0_excel, time),
    concentration)

print(f"The initial plasma concentration with gradient descent: {gradient_descent_parameters["c0"]}")
print(f"The initial plasma concentration with excel: {c0_excel}")
print(f"The elimination rate with gradient descent: {gradient_descent_parameters["k"]}")
print(f"The elimination rate with excel: {k_excel}")
print(f"The cost with gradient descent: {gradient_descent_cost}")
print(f"The cost with excel: {excel_cost}")