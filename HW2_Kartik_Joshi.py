#Homework 2 Kartik Joshi
import random

import numpy as np

# Defining the values of probablities

probs = [0.25, 0.05, 0.2, 0.1, 0.25, 0.15]

# Function for dice rolls


def dice_roll(num_of_rolls, probs):
    rolls = [random.choices(range(1, 7), probs)[0]
             for _ in range(num_of_rolls)]
    return rolls


# First set of 10 rolls

f10_roll = dice_roll(10, probs)

# Probs and frq of first 10 rolls

value10_roll = [f10_roll.count(i) / 10 for i in range(1, 7)]

# Second set of 20 rolls using the previous values

s20_roll = dice_roll(20, value10_roll)

# Values for both sets of rolls

eval10_roll = np.mean(f10_roll)

var10_roll = np.var(f10_roll)

eval20_roll = np.mean(s20_roll)

var20_roll = np.var(s20_roll)

# Expected Results

print("First 10 roll:", f10_roll)

print("Value of probabilities 10 rolls:", value10_roll)

print("Second 20 rolls:", s20_roll)

print("Expected values 10 rolls:", eval10_roll)

print("Variance 10 rolls:", var10_roll)

print("Expected value 20 rolls:", eval20_roll)

print("Variance 20 rolls:", var20_roll)