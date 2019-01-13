
import numpy as np
import keras
import csv
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
import keras.regularizers as regularizers
import matplotlib.pyplot as plt


class BiLSTMSentimentClassifier():
    
    def load_GloVe(self):
        """
        return:
            word_idx of dict type: maps word idx in embedding_wgt matrix to word tokens
            embedding_wgt of torch.float matrix: 50d, pretrained word embedding from GloVE, pre-downloaded in data folder
        """
        with open(r'data\glove\glove.6B.50d.txt', 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            glove = [line for line in reader]

        pad_symbol = '<PAD>'
        unknown_symbol ='<UNK>'
        embedding_wgt = np.array([line[1:] for line in glove]).astype(np.float)
        word_idx = {glove[i][0]:i+1 for i in range(0, len(glove) ) }
        # the <PAD> token is indexed with 0, this will be easier for padding with zero tensors
        word_idx[pad_symbol] = 0
        # the <UNK> token is indexed with last workd
        word_idx[unknown_symbol] = len(glove)+1
        embedding_wgt = np.concatenate((np.zeros((1, 50), dtype=float), embedding_wgt), axis=0)
        embedding_wgt = np.concatenate((embedding_wgt, np.random.rand(1, 50)), axis=0)

        return embedding_wgt, word_idx
    
    def senToken_to_sequence(self,senToken):
        sequence = []
        for word in senToken:
            if word in self.word_idx:
                sequence.append(self.word_idx[word])
            else:
                sequence.append(self.word_idx['<UNK>'])
        return sequence   
    
    def text_to_padded_sequence(self,text):
        sequences = [self.senToken_to_sequence(text_to_word_sequence(sentence, lower=True, split=' ')) for sentence in text]
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sen_len, dtype='int32', padding='post', value=0)
        
        return padded_sequences

    def __init__(self, max_sen_len, hidden_unit, batch_size, epoch, learning_rate, dropout, l2_regularization):
        self.max_sen_len = max_sen_len
        self.hidden_unit = hidden_unit
        self.batch_size = batch_size
        self.epoch=epoch
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.l2_regularization = l2_regularization
        
        self.embedding_wgt, self.word_idx =self.load_GloVe()
        
        self.model = Sequential()
        embedding = Embedding(input_dim=len(self.word_idx), output_dim=50, weights=[self.embedding_wgt], input_length=self.max_sen_len, trainable=False)
        lstm = LSTM(units=self.hidden_unit, activation='relu', use_bias=True, dropout=self.dropout, return_sequences=False)
        dense = Dense(units=1, activation='sigmoid',kernel_regularizer=regularizers.l2(self.l2_regularization))
        self.model.add(embedding)
        self.model.add(Bidirectional(lstm, merge_mode='concat', input_shape=(self.max_sen_len, 1)))
        self.model.add(dense)
        
    def preprocess(self,X,y):
        X = self.text_to_padded_sequence(X)
        y = y.reshape(y.shape[0],1)
        return (X, y)
    
    def fit(self,X, y):
        X, y = self.preprocess(X, y)
        optimizer=Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
        return self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epoch)
        
    def score(self,X,y):
        X, y = self.preprocess(X, y)
        loss, accuracy = self.model.evaluate(X, y)
        return loss, accuracy
    
    def predict(self, X):
        X = self.text_to_padded_sequence(X)
        ypredict = self.model.predict(X)
        return ypredict
    
    def fit_plot(self,X, y, validation_split):
        X, y = self.preprocess(X, y)
        optimizer=Adam(lr=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
        return self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epoch, validation_split=validation_split) 
    
        
        
        



#Test with toxic comment (from kaggle) data set - doesnt really work out good
data = pd.read_csv("data/kaggle_toxic_comment/train.csv", header=0, sep=",", usecols=['comment_text', 'toxic'])
X = data['comment_text'].values
y = data['toxic'].values

classifier = BiLSTMSentimentClassifier(max_sen_len=200, hidden_unit=256, batch_size=32, epoch=5, learning_rate=0.001, dropout=0.5, l2_regularization=0.01)
classifier.fit(X, y)

data = pd.read_csv("data/kaggle_toxic_comment/test.csv", header=0, sep=",")
X = data['comment_text'].values
data = pd.read_csv("data/kaggle_toxic_comment/test_labels.csv", header=0, sep=",")
y = data['toxic'].values
loss, accuracy = classifier.score(X, y)
print('Testing set Accuracy: %f' % (accuracy*100))
print('Testing set loss: %f' % (loss*100))


#Test with yelp comment UCI data set
data = pd.read_csv("data/sentiment_labelled_sentences/yelp_labelled.txt", header=None, sep="\t")
X = data[0].values
y = data[1].values

classifier = BiLSTMSentimentClassifier(max_sen_len=30, hidden_unit=32, batch_size=32, epoch=100, learning_rate=0.001, dropout=0.2, l2_regularization=0.015)
history = classifier.fit_plot(X, y, validation_split=0.2)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

