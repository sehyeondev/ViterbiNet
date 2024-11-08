import numpy as np

def viterbi(priors, n_const, n_mem_size):
    """
    Apply Viterbi detection from computed priors
    """
    data_size = priors.shape[0]
    n_states = n_const**n_mem_size
    x_hat = np.zeros(data_size)
    
    # Generate trellis matrix
    trellis = np.zeros((n_states, n_const), dtype=int)
    for ii in range(n_states):
        idx = ii % (n_const**(n_mem_size-1))
        for ll in range(n_const):
            trellis[ii, ll] = n_const*idx + ll
    
    # Apply Viterbi
    cost = -np.log(priors + 1e-10)  # Add small constant to avoid log(0)
    c_tilde = np.zeros(n_states)
    
    for kk in range(data_size):
        c_tilde_next = np.zeros(n_states)
        for ii in range(n_states):
            temp = np.zeros(n_const)
            for ll in range(n_const):
                temp[ll] = c_tilde[trellis[ii, ll]] + cost[kk, ii]
            c_tilde_next[ii] = np.min(temp)
        c_tilde = c_tilde_next
        i = np.argmin(c_tilde)
        x_hat[kk] = (i % n_const) + 1
    
    return x_hat