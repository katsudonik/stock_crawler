# coding:utf-8
import numpy
import pandas
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

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
        for i in range(len(data) - n_prev):
            X.append(data.iloc[i:(i+n_prev)].as_matrix())
            Y.append(data.iloc[i+n_prev].as_matrix())
        retX = numpy.array(X)
        retY = numpy.array(Y)
        return retX, retY


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

    def learn(self, lstm, year):
        name = self.csv.replace('{{year}}', str(year))
        data_ = pandas.read_csv(name)
        print(str(year))
        data = data_
        if (data is not None):
            pandas.concat([data, data_])
        
        data.columns = ['date', 'open', 'high', 'low', 'close']
        data['date'] = pandas.to_datetime(data['date'], format='%Y-%m-%d')

        data['close'] = preprocessing.scale(data['close'])
        data = data.sort_values(by='date')
        data = data.reset_index(drop=True)
        data = data.loc[:, ['date', 'close']]

        split_pos = int(len(data) * 0.8)
        x_train, y_train = lstm.load_data(data[['close']].iloc[0:split_pos], lstm.length_of_sequences)
        x_test,  y_test  = lstm.load_data(data[['close']].iloc[split_pos:], lstm.length_of_sequences)

        model = lstm.train(x_train, y_train)

        predicted = model.predict(x_test)
        result = pandas.DataFrame(predicted)
        result.columns = ['predict']
        result['actual'] = y_test
        result.plot()
        plt.show()
    
    def run(self):
        lstm = Lstm()

        data = None
        for year in range(2007, 2017):
            self.learn(lstm, year)
            
            
