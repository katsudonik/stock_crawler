# coding:utf-8
import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

from pprint import pprint

import os


os.environ["NLS_LANG"] = "JAPANESE_JAPAN.AL32UTF8"

class Lstm :

    def __init__(self):
        self.length_of_sequences = 5
        self.in_out_neurons = 1
        self.hidden_neurons = 300
        self.batch_size = 5
        self.csv = 'csv/indices_I101_1d_{{year}}.csv'

    def load_data(self, data):
        X, Y = [], []
        for i in range(len(data) - self.length_of_sequences):
            X.append(data.iloc[i:(i+self.length_of_sequences)].as_matrix()) #append 10 rows (past flow)
            Y.append(data.iloc[i+self.length_of_sequences].as_matrix()) #next row of X's 10 rows (predict value)

        for i in range(len(X)):
            self.diplay_sequence(X[i])

        retX = numpy.array(X)
        retY = numpy.array(Y)
        return retX, retY

    def fetch_analyze_data(self, data_):
        data = data_
        if (data is not None):
            pandas.concat([data, data_]) #TODO connect two data (?)-> do nothing...

        #format
        data.columns = ['date', 'open', 'high', 'low', 'close']
        data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
        data['close'] = preprocessing.scale(data['close'])
        data = data.sort_values(by='date')
        data = data.reset_index(drop=True)
        data = data.loc[:, ['date', 'close']] # specificate data's column label(:,)

        #get 'close' data and split it into train/test
        split_pos = int(len(data) * 0.8)
        train = data[['close']].iloc[0:split_pos]
        test  = data[['close']].iloc[split_pos:]

        all_data = {}
        all_data['x_train'], all_data['y_train'] = self.load_data(train)
        all_data['x_test'],  all_data['y_test']  = self.load_data(test)

        return all_data

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
        model = self.create_model()
        model.fit(X_train, y_train, self.batch_size, nb_epoch=100) #default:shuffle=True
        return model

    def display(self, predicted, actual):
        result = pandas.DataFrame(predicted)
        result.columns = ['predict']
        result['actual'] = actual
        result.plot()
        plt.show()

    def diplay_sequence(self, sequence):
        result = pandas.DataFrame(sequence)
        result.columns = ['train_sequence']
        result.plot()
        plt.show()

    def learn(self, year):
        print(str(year))
        name = self.csv.replace('{{year}}', str(year))
        data = self.fetch_analyze_data(pandas.read_csv(name))
        model = self.train(data['x_train'], data['y_train'])
        self.display(model.predict(data['x_test']), data['y_test'])

    def run(self):
        for year in range(2007, 2017):
            self.learn(year)
            
            
