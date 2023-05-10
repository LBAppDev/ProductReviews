import pickle

with open('Network.pickle', 'rb') as f:
    neuralNetwork = pickle.load(f)

def dataProcessing2(comments):
    #tokenizer = Tokenizer()
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    data = tokenizer.texts_to_matrix(comments, mode='tfidf')
    return data

def test():
    again = True
    while again:
        text = input('write a comment: ')
        test_data = dataProcessing2([text])
        prediction = neuralNetwork.forward(test_data[0].reshape(-1, 1))
        print("prediction",prediction[0])

test()