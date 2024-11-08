from sklearn.mixture import GaussianMixture
from tensorflow.keras import layers, models

def get_viterbi_net(input_size, num_hidden_units, num_classes):
    """
    Create LSTM model using TensorFlow
    """
    model = models.Sequential([
        layers.Input(shape=(None, input_size)),
        layers.LSTM(num_hidden_units, 
                   return_sequences=False,
                   kernel_initializer='zeros',
                   recurrent_initializer='zeros'),
        # layers.Dense(num_hidden_units),
        # layers.ReLU(),
        layers.Dense(num_hidden_units//2),
        layers.ReLU(),
        layers.Dense(num_classes),
        layers.Softmax()
    ])
    return model

def get_gaussian_mixture(y_train, n_states):
    """
    Fit Gaussian Mixture Model to data
    """
    
    gmm = GaussianMixture(n_components=n_states, reg_covar=0.1)
    gmm.fit(y_train.reshape(-1, 1))
    return gmm
