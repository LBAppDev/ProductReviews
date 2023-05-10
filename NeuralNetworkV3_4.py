#L1 regularization

import numpy as np

def binray_cross_entropy(y, p):
    e = 1e-5
    output = -(y * np.log(p + e) + (1 - y) * np.log(1 - p + e))
    normalized_output = output
    return normalized_output

def binray_cross_entropy_derivative(y, p):
    e = 1e-5
    return (p - y ) / ((p + e) * (1 - p + e))

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(output_size,input_size)
        self.bias = np.zeros((output_size,1))
        self.activation = activation
        self.inputs = None
        self.outputs = None
        self.input_size = input_size
        self.output_size = output_size

        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)

        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

    def forward(self, inputs):
        self.inputs = inputs
        z = np.dot(self.weights, self.inputs) + self.bias
        self.outputs = self.activation(z)
        return self.outputs
    
    def backward(self, delta):
        activation_derivative = self.activation(self.outputs, derivative=True)
        self.delta = delta * activation_derivative
        input_delta = np.dot(self.weights.T, self.delta)
        return input_delta
    
    def update_weights(self, learning_rate, beta1, beta2, epsilon, t, l1_lambda = 0):
        dW = np.dot(self.delta, self.inputs.T)
        dB = np.sum(self.delta, axis=1, keepdims=True)

        l1_term = l1_lambda * np.sign(self.weights)

        #update weights
        self.m_w = beta1 * self.m_w + (1 - beta1) * dW
        self.v_w = beta2 * self.v_w + (1 - beta2) * (dW ** 2)

        m_w_hat = self.m_w / (1 - beta1 ** t)
        v_w_hat = self.v_w / (1 - beta2 ** t)
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon) + l1_term

        #update bias
        self.m_b = beta1 * self.m_b + (1 - beta1) * dB
        self.v_b = beta2 * self.v_b + (1 - beta2) * (dB ** 2)

        m_b_hat = self.m_b / (1 - beta1 ** t)
        v_b_hat = self.v_b / (1 - beta2 ** t)
        self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
    
class NeuralNetwork:
    def __init__(self):
        self.hidden_layers = []

    def forward(self, inputs):
        hidden_outputs = inputs
        if len(self.hidden_layers) != 0:
            for hidden_layer in self.hidden_layers:
                hidden_outputs = hidden_layer.forward(hidden_outputs)
        return hidden_outputs
    
    def backward(self, y, output):
        output_delta = binray_cross_entropy_derivative(y, output)
        if len(self.hidden_layers) != 0:
            self.hidden_backward(output_delta)

    def update_weights(self, learning_rate, beta1, beta2, epsilon, t, l1_lambda = 0):

        for layer in self.hidden_layers:
            layer.update_weights(learning_rate, beta1, beta2, epsilon, t, l1_lambda)
        
    def train(self, x, y, epochs = 7, learining_rate = 0.1, batch_size = 32, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, l1_lambda = 0):

        num_batches = len(x) // batch_size

        for i in range(epochs):
            loss = 0
            correct = 0

            indices = np.random.permutation(len(x))
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for j in range(num_batches):

                start_index = j * batch_size
                end_index = start_index + batch_size
                x_batch = x_shuffled[start_index:end_index]
                y_batch = y_shuffled[start_index:end_index]

                #forward pass
                output = self.forward(x_batch.T)
                
                loss += binray_cross_entropy(y_batch, output)
                l1_loss = l1_lambda * self.calculate_l1_loss(l1_lambda)
                loss += l1_loss
                #calculating accuracy
                predictions = np.where(output > 0.5, 1, 0)
                correct += np.sum(predictions == y_batch)
                self.backward(y_batch, output)

                #update weights using adam
                t = i *  num_batches + j + 1 # time step

                #update weights
                self.update_weights(learining_rate, beta1, beta2, epsilon, t, l1_lambda)

            avr_loss = np.mean(loss) / num_batches
            accuracy = correct / (num_batches * batch_size)
            print("epoch {}, loss: {}, accuracy: {}".format(i+1, avr_loss, accuracy))
        return correct/len(x)

    def predict(self, x):
        output = self.forward(x)
        return np.round(output)
    
    def add_layer(self, layer):
        self.hidden_layers.append(layer)

    def hidden_backward(self, delta):
        hidden_delta = delta
        for hidden_layer in reversed(self.hidden_layers):
            hidden_delta = hidden_layer.backward(hidden_delta)

    def calculate_l1_loss(self, l1_lambda = 0):
        l1_loss = 0
        for layer in self.hidden_layers:
            l1_loss += l1_lambda * np.sum(np.abs(layer.weights))
        return l1_loss