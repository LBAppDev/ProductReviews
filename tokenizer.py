print("importing modules ...")

import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle

print("importing modules complete !")

print("reading the data ...")

df = pd.read_csv('test.csv') # reading data set

print("reading the data complete !")

inputLayer = 512

tokenizer = Tokenizer(num_words=inputLayer) # convert words to numbers
def dataProcessing(comments):
    global tokenizer
    tokenizer.fit_on_texts(comments)
    data = tokenizer.texts_to_matrix(comments, mode='tfidf')
    return data

x = df['review_text']
df["feedback"] -= 1
y = df["feedback"]

print("data processing ...")

x = dataProcessing(x)
print(x.shape)

print("data processing complete !")

with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

print("tokenizer save !")