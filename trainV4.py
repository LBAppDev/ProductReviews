print("importing modules ...")
from NeuralNetworkV3_4 import NeuralNetwork
from NeuralNetworkV3_4 import Layer
from Neuron import Neuron
import pickle
import numpy as np

print("importing modules complete !")

print("Creating the model ...")

inputLayer = 512

hidden = 8

output = 1

hidden_activation = Neuron.sigmoid
input_activation = Neuron.sigmoid
output_activation = Neuron.sigmoid

neuraletwork = NeuralNetwork()

l1 = Layer(inputLayer, hidden, hidden_activation)
neuraletwork.add_layer(l1)

l2 = Layer(hidden, hidden, hidden_activation)
neuraletwork.add_layer(l2)

l3 = Layer(hidden, hidden, hidden_activation)
neuraletwork.add_layer(l3)

l4 = Layer(hidden, hidden, hidden_activation)
neuraletwork.add_layer(l4)

l5 = Layer(hidden, hidden, hidden_activation)
neuraletwork.add_layer(l5)

output_layer = Layer(hidden, output, output_activation)
neuraletwork.add_layer(output_layer)




print("model complete !")

def dataProcessing2(comments):
    #tokenizer = Tokenizer()
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    data = tokenizer.texts_to_matrix(comments, mode='tfidf')
    print(data.shape)
    return data

with open('x_train.pickle', 'rb') as f:
    x_train = pickle.load(f)

with open('y_train.pickle', 'rb') as f:
    y_train = pickle.load(f)


#training the model
print("training ...")
y_train = y_train.values

x_train_normalized = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))

max_accuracy = 0
max_learning_rate = 0

for i in range(50, 100):
    accuracy = neuraletwork.train(x_train_normalized, y_train, epochs = 10,learining_rate = 0.1 / i, batch_size = 32, l1_lambda = 0.0001)
    if(accuracy > max_accuracy):
        max_accuracy = accuracy
        max_learning_rate = i
    print("max accuracy: ", max_accuracy)
    print("max learning rate: ", max_learning_rate)
    print("current: ", i)

print("training complete !")
# some good rates relu =  0.02 
#                 after batch [0.1 / 4 = 80] [0.1 / 8 pretty close] [ 0.1 / 9 b = 32]
#                 simgoid = 0.05 0.002 0.004 0.006 [0.025 = 80] [0.0125 / 2 ** 5 = 79]
#                 [0.0125 / 2 ** 3 = 80] [0.0125 / 2 ** 2.1 = 80] [0.0125 / 2 ** 1.5 = 80]
#                 [0.0125 / 2 ** 1.255 = 80]
#                 after bath [0.1 / 2 ** 4.119 = 80] [0.1 / 2 ** 4.2 = 80.3]

with open('Network.pickle', 'wb') as f:
    pickle.dump(neuraletwork, f)

print("model saved !")