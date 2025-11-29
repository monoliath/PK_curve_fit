import numpy as np
from gradient_descent import get_file, forward_propagation_1comp, compute_cost, gradient_descent_1comp
from utilis import plot_data_1comp

# Get the data
data = get_file()

# Get the time and concentration
time = np.array(data.iloc[:, 0])
time = time.reshape(1, len(time))
concentration = np.array(data.iloc[:, 1])
concentration = concentration.reshape(1, len(concentration))

# Gradient descent model
gradient_descent_parameters = gradient_descent_1comp(data)

c0_gd = gradient_descent_parameters["c0"]
k_gd = gradient_descent_parameters["k"]

gradient_descent_cost = compute_cost(
    forward_propagation_1comp(c0_gd, k_gd, time),
    concentration)

# Excel model
c0_excel = float(input("Enter your excel calculation of the initial plasma concentration:"))
k_excel = float(input("Enter your excel calculation of the elimination rate:"))

excel_cost = compute_cost(
    forward_propagation_1comp(c0_excel, k_excel, time),
    concentration)

print("\n\nGradient descent calculation:")
print(f"The initial plasma concentration with gradient descent: {gradient_descent_parameters["c0"]}")
print(f"The elimination rate with gradient descent: {gradient_descent_parameters["k"]}")
print(f"The cost with gradient descent: {gradient_descent_cost}")

print("\nExcel calculation:")
print(f"The initial plasma concentration with excel: {c0_excel}")
print(f"The elimination rate with excel: {k_excel}")
print(f"The cost with excel: {excel_cost}")

plot_data_1comp(data, c0_gd, k_gd, c0_excel, k_excel)