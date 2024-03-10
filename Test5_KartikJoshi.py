# -*- coding: utf-8 -*-
"""
Test-05
Permutation test
"""

import numpy as np

# We are using pre defined user data to feed for this model
groupA = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
groupB = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

#Finding the diffrence of the avg of the groups
mean_A = np.mean(groupA)
mean_B = np.mean(groupB)
initial_difference = abs(mean_A - mean_B)

print(initial_difference)

#user defined number of permutations(we are taking 1000 )
num_perms = 1000

#a time counter
time_count = 0

#outer loop to perform series of test
for _ in range(num_perms):
    
    combined_data = np.concatenate((groupA, groupB))
    #combining the data
    
    np.random.shuffle(combined_data)
    #suffling the combination of data
    
    permuted_Ga = combined_data[:len(groupA)]
    permuted_Gb = combined_data[len(groupA):]
    #spiliting the data
    
    permuted_difference = abs(np.mean(permuted_Ga) - np.mean(permuted_Gb))

print(permuted_difference)

#Find if permuted difference is greater than initial difference
if permuted_difference > initial_difference:
        time_count += 1

#Finding the p value for the given experiment 
p_val = time_count / num_perms

print(f"Initial absolute difference of averages: {initial_difference}")
print(f"Permutted difference of averages: {permuted_difference}")
print(f"P-value: {p_val}")