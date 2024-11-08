# %%
import numpy as np
import tensorflow as tf
from sklearn.mixture import GaussianMixture
from helpers import my_reshape
from Viterbi import viterbi
from ViterbiNet import get_viterbi_net, get_gaussian_mixture

# Constants and parameters
n_const = 2        # Constellation size (2 = BPSK)
n_mem_size = 4     # Number of taps
train_size = 500   # Training size
test_size = 10   # Test data size
channel_exp = 0.5    # Exponential decay factor
sigma_w = 10**(-0.2)  # Noise variance of LTI AWGN channel
n_states = n_const**n_mem_size

# Generate channel
channel = np.exp(-channel_exp * np.arange(n_mem_size))

# %%
# Generate training labels
x_train = np.random.randint(1, n_const + 1, size=train_size)
s_train = 2 * (x_train - 0.5*(n_const + 1))
m_x_train = my_reshape(x_train, n_mem_size)
m_s_train = my_reshape(s_train, n_mem_size)
r_train = np.flip(channel) @ m_s_train
y_train = r_train + np.sqrt(sigma_w) * np.random.randn(*r_train.shape)

# Generate test labels
x_test = np.random.randint(1, n_const + 1, size=test_size)
s_test = 2 * (x_test - 0.5*(n_const + 1))
m_s_test = my_reshape(s_test, n_mem_size)
r_test = np.flip(channel) @ m_s_test
# y_test = r_test + np.sqrt(sigma_w) * np.random.randn(*r_test.shape)
y_test = np.array([-2.22164873378819, -1.82916271010549, -1.62082334403308, 0.332987709676881, 1.70937654040036, -0.361017479843602, -0.770364202050401, 0.783350004589344, 1.89571736003345, 1.67403378690826])

# %%
# Neural network parameters
input_size = 1
num_hidden_units = 100
num_classes = n_states

# Create and compile model
model = get_viterbi_net(input_size, num_hidden_units, num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Format training data
combine_vec = n_const ** np.arange(n_mem_size)
x_cat = np.dot(combine_vec, (m_x_train - 1))
y_train_reshaped = y_train.reshape(-1, 1, 1)

# Train model
history = model.fit(
    y_train_reshaped, 
    x_cat,
    epochs=100,
    batch_size=27,
    verbose=1
)
# %%
# Predict using trained model
y_test_reshaped = y_test.reshape(-1, 1, 1)
p_s_y = model.predict(y_test_reshaped)

# %%
# Compute output PDF using GMM fitting
gmm = GaussianMixture(n_components=n_states, reg_covar=0.1)
gmm.fit(y_train.reshape(-1, 1))
p_y = np.exp(gmm.score_samples(y_test.reshape(-1, 1)))

# %%
# Compute likelihoods
likelihood = (p_s_y * p_y.reshape(-1,1)) * n_states

# %%
# Apply Viterbi output layer
x_hat = viterbi(likelihood, n_const, n_mem_size)

# %%
# Evaluate error rate
ser = np.mean(x_hat != x_test)
print(f"Symbol Error Rate: {ser}")
# %%
