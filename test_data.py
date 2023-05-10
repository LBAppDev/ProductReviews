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

def dataProcessing(comments):
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    data = tokenizer.texts_to_matrix(comments, mode='tfidf')
    print(data.shape)
    return data

x = df['review_text']
df["feedback"] -= 1
y = df["feedback"]

print("data processing ...")

x = dataProcessing(x)

print("data processing complete !")

print("tokenizer save !")
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state =42)
with open('x_train.pickle', 'wb') as f:
    pickle.dump(x_train, f)

with open('y_train.pickle', 'wb') as f:
    pickle.dump(y_train, f)
print("data saved !")