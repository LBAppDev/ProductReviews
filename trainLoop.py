import re
from nltk.tokenize import word_tokenize # for tokenization
from nltk.stem import PorterStemmer # for stemming
from nltk.corpus import stopwords
import pandas as pd
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import array 

df = pd.read_csv('/home/ibrahim/PycharmProjects/AmazonProductReviewsEvaluation/amazon_alexa.tsv',sep='\t')
#data processing function
def data_processing(text):
    text = text.lower()
    text = re.sub(r"http\S+www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

#applying data processing
df.verified_reviews = df['verified_reviews'].apply(data_processing)

#data stemming
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

#applying data stemming
df['verified_reviews'] = df['verified_reviews'].apply(lambda x: stemming(x))

x = df['verified_reviews']
y = df['feedback']

#converting words to numbers
cv = CountVectorizer()
x = cv.fit_transform(df['verified_reviews'])

#spliting to training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.toarray()
x_test = x_test.toarray()

from keras.models import Sequential
from keras.layers import Dense

max_accuracy = 0
max_model = None
layers = array.array('i', [0]*10)
max_layers = array.array('i',[0] *10)

def nested_loop(max_depth, max_nodes, index = 0, model = Sequential()):
    global max_accuracy
    global max_model
    global layers
    global max_layers
    if(index == max_depth):
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=10, epochs=10)

        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        if(test_accuracy > max_accuracy):
            max_accuracy = test_accuracy
            max_model = model
            max_layers = array.array('i', [0] * len(layers))
            for k in range (len(max_layers)):
                 max_layers[k] = layers[k]
        print('Test loss: ',test_loss)
        print('Test accuracy: ', test_accuracy)
        print('max accuracy: ', max_accuracy)
        print('max hidden layers: ', len(max_layers))
        for i in range(len(max_layers)):
             print('max layer[', i,'] = ', max_layers[i])
        print('current hidden layers: ', len(layers))
        for i in range(len(layers)):
             print('layer[', i,'] = ', layers[i])
        model.pop()
    else :
        for i in range(1, max_nodes+1):
            model.add(Dense(units=i, activation='relu'))
            layers[index] = i
            nested_loop(max_depth, max_nodes, index + 1, model)
            model.pop()
                
for i in range(1, 16):
        layers = array.array('i', [0] * (i))
        nested_loop(i,32)