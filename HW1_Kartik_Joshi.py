"""
HW1 
Markov Inequality
"""

import numpy as np
import pandas as pd

# Using the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_name = ["sepal_length", "sepal_width",
            "petal_length", "petal_width", "class"]
iris_data = pd.read_csv(url, header=None, names=col_name)

# Setting the value of X
x = 10

# Randomly selecting the X values from the 'petal_length' column
selected_vals = iris_data["petal_length"].sample(n=x, random_state=42)

# Calculating the expected value
expected_vals = selected_vals.mean()

# Defining the function for Markov Inequality


def markovs(a, expected_vals):
    probab = expected_vals / a
    return probab


# Applying Markov's Inequality given dataset
for a_value in selected_vals:
    probability_of_a = markovs(a_value, expected_vals)
    print(
        f"For a = {a_value}, Probability(P(petal_length >= {a_value})) ≤ {probability_of_a}")

# Select the value of a
a = 4.0

# Calculating for petal_length >= a
actual_probabs = (iris_data["petal_length"] >= a).mean()

# Calculating the expected value (mean) for petal_length
expected_vals = iris_data["petal_length"].mean()

# Calculating the upper bound
markov_upper_bound = expected_vals / a

# Displaying the results
print(f"Actual Probability(P(petal_length >= {a})) = {actual_probabs}")
print(f"Markov's Inequality Upper Bound = {markov_upper_bound}")

# Selecting the value a
a = 4.0

# Defining the function given a value and a series of data


def markovs(a, data_series):
    expected_vals = data_series.mean()
    probab = expected_vals / a
    return probab


# Applig Markov's Inequality for each feature
for column in iris_data.columns[:-1]:
    probab = markovs(a, iris_data[column])
    print(
        f"Feature: {column}, Actual Probability(P({column} >= {a})) ≤ {probab}")

# Setting the value of x
x = 10

# Randomly selecting the values of x for petal_length column
selected_vals = iris_data["petal_length"].sample(n=x, random_state=42)

# Calculating the expected value
expected_vals = selected_vals.mean()

# Create a dataFrame for results
result_dataframe = pd.DataFrame({
    "Selected Values": selected_vals,
    "Expected Values": [expected_vals] * x,
    "Markov Inequality Upper Bounds": expected_vals / selected_vals
})

# Displaying the results
print(result_dataframe)


# Creating dataset of single random variable X
data = {'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}


df = pd.DataFrame(data)

# Setting the value of x
x = 5

# Randomly select value of x for the 'X' column
selected_vals = df['X'].sample(n=x, random_state=42)

# Calculate the expected value (mean) for the selected values
expected_vals = selected_vals.mean()

# Calculating Markov Inequality for each selected value
markov_values = expected_vals / selected_vals

# Create a DataFrame for results
result_dataframe = pd.DataFrame({
    'Selected Values': selected_vals,
    'Expected Values': [expected_vals] * x,
    'Markov Inequality Values': markov_values
})

# Displaying the results
print(result_dataframe)


"""
HW1 : Part 2 
Chebyshev Inequality
"""


# Using the same data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_name = ["sepal_length", "sepal_width",
            "petal_length", "petal_width", "class"]
iris_data = pd.read_csv(url, header=None, names=col_name)

# Randomly selecting the X values from the 'petal_length' column
selected_features = "petal_length"

# Choose a specific value 'k' (number of standard deviations)
k = 2

# Calculating the mean and standard deviation
mean = iris_data[selected_features].mean()
std_dev = iris_data[selected_features].std()

# Calculating the upper bound
probab = 1 - 1 / (k**2)

# Result
print(
    f"Chebyshev's Inequality: Probability(P(|{selected_features} - μ| ≥ {k}σ)) ≤ {probab}")


# Creating dataset of single random variable X
data = {'X': [5, 10, 15, 20, 25, 30, 35, 40, 45,
              50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

df = pd.DataFrame(data)

# List of X values to for Chebyshev's Inequality
x_vals = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

# Create DataFrame for results
result_dataframe = pd.DataFrame(
    columns=['x', 'Expected Value', 'Variance', 'Chebyshev Bound'])

# Calculate expected value and variance for each X value
for x in x_vals:
    expected_vals = df['X'].mean()
    variance = df['X'].var()

    # Calculate the upper bound
    k = abs((x - expected_vals) / variance)
    chebyshev_bound = 1 / (k**2)

    result_dataframe = result_dataframe.append({'x': x,
                                                'Expected Value': expected_vals,
                                                'Variance': variance,
                                                'Chebyshev Bound': chebyshev_bound},
                                               ignore_index=True)

# Displaying the results
print(result_dataframe)
