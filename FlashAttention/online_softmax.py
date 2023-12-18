import numpy as np
import pdb
pdb.set_trace()

def naive_softmax(src):
    max_value = np.max(src)
    exps = np.exp(src - max_value)
    return exps / np.sum(exps)


def fast_softmax(src):
    breakpoint()
    old_max = -np.inf
    sum_exp = 0.0
    for value in src:
        new_max = max(old_max, value)
        sum_exp = sum_exp * np.exp(old_max - new_max) + np.exp(value - new_max)
        old_max = new_max

    exps = np.exp(src - old_max)
    return exps / sum_exp

'''
# Test with an array of length 100,000
data_len = 100000
test_array = np.random.rand(data_len)

# Apply both softmax functions
naive_result = naive_softmax(test_array)
fast_result = fast_softmax(test_array)

# Verify if the results are similar
np.allclose(naive_result, fast_result), naive_result[:5], fast_result[:5]  # Displaying the first 5 elements of each result for comparison
'''

test_data = np.array([1, 2, 3, 4])
print(fast_softmax(test_data))
