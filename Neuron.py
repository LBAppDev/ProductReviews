import numpy as np
import warnings

class Neuron:

    def sigmoid(unactivated_output, derivative = False):
        if derivative:
            return Neuron.sigmoid(unactivated_output) * (1 - Neuron.sigmoid(unactivated_output))
        activated_output = 1 / (1 + np.exp(-unactivated_output))
        return activated_output

    
    def relu(z, derivative = False, clip_threshod = 10000):
        if derivative:
            return np.where(z > 0, 1, 0)
        return np.clip(z, 0, clip_threshod)
