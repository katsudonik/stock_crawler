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
        self.length_of_sequences = 10
        self.in_out_neurons = 1
        self.hidden_neurons = 300
        self.csv = 'csv/indices_I101_1d_{{year}}.csv'

    def load_data(self, data, n_prev=10):
        X, Y = [], []
        pprint(vars(data))
        for i in range(len(data) - n_prev):
            print(i)
            X.append(data.iloc[i:(i+n_prev)].as_matrix()) #TODO speficicate row
            Y.append(data.iloc[i+n_prev].as_matrix()) #TODO
        retX = numpy.array(X)
        retY = numpy.array(Y)
        return retX, retY

    def fetch_analyze_data(self, data_, analyze_column = ['date', 'close']):
        data = data_
        if (data is not None):
            pandas.concat([data, data_]) #TODO connect two data (?)

        #format
        data.columns = ['date', 'open', 'high', 'low', 'close']
        data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')
        data['close'] = preprocessing.scale(data['close'])
        data = data.sort_values(by='date')
        data = data.reset_index(drop=True)
        data = data.loc[:, analyze_column] # specificate data's column label(:,)

        #split train/test by close?
        split_pos = int(len(data) * 0.8)
        data['train'] = data[['close']].iloc[0:split_pos]
        data['test']  = data[['close']].iloc[split_pos:]

        all_data = {}
        all_data['x_train'], all_data['y_train'] = self.load_data(data['train'], self.length_of_sequences)
        all_data['x_test'],  all_data['y_test']  = self.load_data(data['test'], self.length_of_sequences)

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
        model.fit(X_train, y_train, batch_size=10, nb_epoch=100)
        return model

    def display(self, predicted, y_test):
        result = pandas.DataFrame(predicted)
        result.columns = ['predict']
        result['actual'] = y_test
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
            
            
