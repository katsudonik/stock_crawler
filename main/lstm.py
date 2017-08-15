# coding:utf-8
import numpy
import pandas
import matplotlib.pyplot as plt
import os

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from pprint import pprint
from numpy.random import *

os.environ["NLS_LANG"] = "JAPANESE_JAPAN.AL32UTF8"

# predict price of next day from closing price up to previous day (20days:1 month)
# train by data of 1 years sequences
# Do not consider the long-term viewpoint

# usage:
# import lstm
# ls=lstm.Lstm('indices/I101')
# ls.run()

class Lstm :

    def __init__(self, relative_url):
        self.length_of_sequences = 20 # predict from previous 20day's value
        self.in_out_neurons = 1 # predict target:stock value
        self.hidden_neurons = 300
        self.batch_size = 10
        self.nb_epoch = 100 # Get data randomly and learn 100 times
        self.csv = 'csv/' + relative_url.replace('/', '_') + '_1d_{{year}}.csv'
        self.display_train_sequence = False

        self.columns = ['date', 'open', 'high', 'low', 'close']
        if 'stock' in relative_url:
            self.columns = ['date', 'open', 'high', 'low', 'close', 'yield', 'sales_value']

        self.model_file = './lstm_model.json'
        self.weights_file = './' + relative_url.replace('/', '_') + '_model.hdf5'

    def load_data(self, data):
        X, Y = [], []
        for i in range(len(data) - self.length_of_sequences):
            X.append(data.iloc[i:(i+self.length_of_sequences)].as_matrix()) #append 10 rows (past flow)
            Y.append(data.iloc[i+self.length_of_sequences].as_matrix()) #next row of X's 10 rows (predict value)

        if(self.display_train_sequence):
            for i in range(len(X)):
                self.diplay_sequence(X[i], ['train_sequence'])

        return numpy.array(X), numpy.array(Y)

    def fetch_analyze_data(self, data_, year):
        data = data_
        if (data is not None):
            pandas.concat([data, data_]) #TODO connect two data (?)-> do nothing...

        #format
        data.columns = self.columns
        data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
        data['close'] = preprocessing.scale(data['close'])
        data = data.sort_values(by='date')
        data = data.reset_index(drop=True)
        data = data.loc[:, ['date', 'close']] # specificate data's column label(:,)

        self.display_all_data(data[['close']], year)
        return self.load_data(data[['close']]) #get 'close' data

    # test term shouldn't be immobilized and test data shouldn't be duplicated with train data
    def divide_into_train_test(self, seq, label):
        num_all = int(len(seq))
        num_test = int(num_all * 0.2)
        pos = randint(num_all - num_test)
        pos_limit = pos + num_test
        list = range(pos, pos_limit)

        xTest = seq[pos : pos_limit]
        xTrain = numpy.delete(seq, list, 0)

        yTest = label[pos : pos_limit]
        yTrain = numpy.delete(label, list, 0)

        return xTrain, xTest, yTrain, yTest

    def create_model(self) :
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, \
                       batch_input_shape=(None, self.length_of_sequences, self.in_out_neurons), \
                       return_sequences=False))
        model.add(Dense(self.in_out_neurons))
        model.add(Activation("linear"))
        model.compile(loss="mape", optimizer="adam")
        return model

    def train(self, X_train, y_train) :
        self.model.fit(X_train, y_train, self.batch_size, nb_epoch=self.nb_epoch) #default:shuffle=True
        return self.model

    def save_model(self):
        open(self.model_file, 'w').write(self.model.to_json())
        self.model.save_weights(self.weights_file)

    def display(self, predicted, actual):
        result = pandas.DataFrame(predicted)
        result.columns = ['predict']
        result['actual'] = actual
        result.plot()
        plt.show()

    def diplay_sequence(self, sequence, columns):
        result = pandas.DataFrame(sequence)
        result.columns = columns
        result.plot()
        plt.show()

    def display_all_data(self, data, year):
        Y = []
        for i in range(len(data)):
            Y.append(data.iloc[i].as_matrix())
        self.diplay_sequence(Y, [str(year)])

    def learn(self, year):
        name = self.csv.replace('{{year}}', str(year))
        sequences, labels = self.fetch_analyze_data(pandas.read_csv(name, encoding="SHIFT-JIS"), year)
        xTrain, xTest, yTrain, yTest = self.divide_into_train_test(sequences, labels)
        model = self.train(xTrain, yTrain)
        self.display(model.predict(xTest), yTest)

    # Every year the market can not be the same. so, don't refresh model through learning all data
    def run(self):
        self.model = self.create_model()
        for year in range(2007, 2017):
            self.learn(year)

        self.save_model()
            

