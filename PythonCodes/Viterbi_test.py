# %%
import numpy as np
from Viterbi import viterbi

def verify_probability_sum(priors):
    """Verify that probabilities sum to approximately 1 for each time instance"""
    sums = np.sum(priors, axis=1)
    if not np.allclose(sums, 1, rtol=1e-10):
        print("Warning: Probabilities do not sum to 1 for all time instances")
        print("Sums:", sums)

def verify_output_range(x_hat, n_const):
    """Verify that all outputs are within the valid range"""
    if np.any(x_hat < 1) or np.any(x_hat > n_const):
        print("Warning: Output contains invalid symbol indices")
        print(f"Range should be: 1 to {n_const}")
        print("Actual values:", x_hat)

# def test_viterbi_detector():
"""Test suite for Viterbi detector implementation"""
# %%
# Test Case 1: Simple case with known output
print("Running Test Case 1: Simple case")
n_const = 2      # Binary constellation
n_mem_size = 1   # Memory length of 1
data_size = 5    # 5 time instances

# Create test priors (probabilities should sum to 1 for each time instance)
priors = np.array([
    [0.9, 0.1],   # Strong probability of state 1
    [0.1, 0.9],   # Strong probability of state 2
    [0.9, 0.1],   # Strong probability of state 1
    [0.1, 0.9],   # Strong probability of state 2
    [0.9, 0.1]    # Strong probability of state 1
])

x_hat = viterbi(priors, n_const, n_mem_size)
print("Expected output: [1 2 1 2 1]")
print("Actual output:  ", x_hat)
np.testing.assert_array_almost_equal(x_hat, np.array([1, 2, 1, 2, 1]))

# Test Case 2: Longer sequence
print("\nRunning Test Case 2: Longer sequence")
data_size = 10
# Create alternating probabilities
priors = np.zeros((data_size, 2))
for i in range(data_size):
    if i % 2 == 0:
        priors[i] = [0.9, 0.1]
    else:
        priors[i] = [0.1, 0.9]

x_hat = viterbi(priors, n_const, n_mem_size)
print("Expected output: [1 2 1 2 1 2 1 2 1 2]")
print("Actual output:  ", x_hat)
np.testing.assert_array_almost_equal(x_hat, np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2]))

# Test Case 3: Larger constellation
print("\nRunning Test Case 3: Larger constellation")
n_const = 4      # 4-symbol constellation
n_mem_size = 2   # Memory length of 2
data_size = 5

# Create priors for 4-symbol constellation
n_states = n_const**n_mem_size
priors = np.ones((data_size, n_states)) * 0.1
# Set strong probabilities for a specific pattern
for i in range(data_size):
    priors[i, (i+1) % 4 ] = 0.7  # Strong probability for symbol i mod 4

x_hat = viterbi(priors, n_const, n_mem_size)
print("Expected pattern should follow 1,2,3,4,1")
print("Actual output:  ", x_hat)

# Verify results using helper functions
verify_probability_sum(priors)
verify_output_range(x_hat, n_const)



def test_edge_cases():
    """Additional test cases for edge cases"""
    
    # Test Case 4: Very small probabilities
    print("\nRunning Test Case 4: Very small probabilities")
    n_const = 2
    n_mem_size = 1
    priors = np.array([
        [0.999999, 0.000001],
        [0.000001, 0.999999]
    ])
    x_hat = viterbi(priors, n_const, n_mem_size)
    print("Expected output: [1 2]")
    print("Actual output:  ", x_hat)
    
    # Test Case 5: Equal probabilities
    print("\nRunning Test Case 5: Equal probabilities")
    priors = np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ])
    x_hat = viterbi(priors, n_const, n_mem_size)
    print("Output with equal probabilities:", x_hat)
    verify_output_range(x_hat, n_const)

# if __name__ == "__main__":
#     # Run all tests
#     test_viterbi_detector()
#     # test_edge_cases()
# %%
