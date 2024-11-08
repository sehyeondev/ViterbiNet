# %%
import csv
import numpy as np
import tensorflow as tf
from ViterbiNet import get_viterbi_net, get_gaussian_mixture

n_const = 2        # Constellation size (2 = BPSK)
n_mem_size = 4     # Number of taps
n_states = n_const**n_mem_size

with open ('y_train.csv', newline='') as csvfile:
    f = csv.reader(csvfile)
    for row in f:
        y_train = np.array([float(i) for i in row])

m_x_train = np.zeros((4, 500))
with open('m_x_train.csv', newline='') as csvfile:
    f = csv.reader(csvfile)
    for i, row in enumerate(f):
        m_x_train[i] = np.array([float(j) for j in row])

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
