import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# Load the Diabetes dataset
diabetes_data = load_diabetes()  # Loading the diabetes dataset
# Select a single feature (column 2) for simplicity
X_data = diabetes_data.data[:, np.newaxis, 2]
Y_data = diabetes_data.target

# Compute coefficients manually
X_mean_value = np.mean(X_data)  # Compute the mean of the input feature
Y_mean_value = np.mean(Y_data)  # Compute the mean of the target variable
slope_manual = np.sum((X_data - X_mean_value) * (Y_data - Y_mean_value)) / \
    np.sum((X_data - X_mean_value) ** 2)  # Calculate the slope
intercept_manual = Y_mean_value - slope_manual * \
    X_mean_value  # Calculate the intercept

# Create the linear equation manually
# Form the equation in string format
linear_equation_manual = f"Manually: Y = {intercept_manual:.2f} + {slope_manual:.2f} * X"

# Calculate predicted values manually
Y_predicted_manual = intercept_manual + slope_manual * \
    X_data  # Calculate the predicted values

# Calculate residuals manually
residuals_manual_calculation = Y_data - \
    Y_predicted_manual  # Compute the residuals

# Calculate RSS manually
# Compute the Residual Sum of Squares (RSS)
RSS_manual_calculation = np.sum(residuals_manual_calculation ** 2)

# Use scikit-learn to compute coefficients
model_sklearn = LinearRegression()  # Initialize the Linear Regression model
model_sklearn.fit(X_data, Y_data)  # Fit the model with the data
intercept_sklearn = model_sklearn.intercept_  # Get the intercept from the model
slope_sklearn = model_sklearn.coef_[0]  # Get the slope from the model

# Create the linear equation using scikit-learn
# Form the equation in string format
linear_equation_sklearn = f"Scikit-Learn: Y = {intercept_sklearn:.2f} + {slope_sklearn:.2f} * X"

# Use scikit-learn to compute RSS
model_sklearn.fit(X_data, Y_data)  # Fit the model again with the data
# Predict the target variable using the model
Y_predicted_sklearn = model_sklearn.predict(X_data)

# Calculate residuals using scikit-learn
# Compute the residuals using scikit-learn
residuals_sklearn_calculation = Y_data - Y_predicted_sklearn

# Calculate RSS using scikit-learn
# Compute the RSS using scikit-learn
RSS_sklearn_calculation = np.sum(residuals_sklearn_calculation ** 2)

# Compare coefficients
print("Manual Coefficients:")
print("Slope_manual:", slope_manual)  # Print manually calculated slope
# Print manually calculated intercept
print("Intercept_manual:", intercept_manual)

print("\nScikit-Learn Coefficients:")
print("Slope_sklearn:", slope_sklearn)  # Print slope from scikit-learn
# Print intercept from scikit-learn
print("Intercept_sklearn:", intercept_sklearn)

# Compare the linear equations
# Print manually created linear equation
print("\nLinear Equation (Manual):", linear_equation_manual)
# Print linear equation from scikit-learn
print("Linear Equation (Scikit-Learn):", linear_equation_sklearn)

# Compare RSS
print("\nManual RSS:", RSS_manual_calculation)  # Print manually calculated RSS
# Print RSS calculated using scikit-learn
print("Scikit-Learn RSS:", RSS_sklearn_calculation)
