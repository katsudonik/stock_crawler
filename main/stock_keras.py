from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import yaml

### functions ###########################################################################
def load_config(path):
    f = open(path, 'r')
    config = yaml.load(f)
    f.close()
    return config

def load_data(quantified_data_path):
    data = {}
    # 分類対象画像を数値化したデータをロード(result of stock-makedata.py) --- (※1)
    X_train, X_test, y_train, y_test = np.load(quantified_data_path)
    # データを0-1に正規化する
    data['X_train'] = X_train.astype("float") / 256 #色は最大で256画素->256で割れば0-1になる
    data['X_test']  = X_test.astype("float")  / 256
    data['y_train'] = np_utils.to_categorical(y_train, len(config['categories'])) #transform to one-hot encoding
    data['y_test']  = np_utils.to_categorical(y_test, len(config['categories']))
    return data

# モデルを構築 --- (※2)
def build_model():
    model = Sequential()

    #keras2->Conv2D
    model.add(Conv2D(32, (3, 3),#filter(convolution window/kernel):3*3size, 32 sheet
    padding='same',
	input_shape=(config['img']['height'], config['img']['width'], config['img']['channels']))) #channels_last #first layer in a model, provide input_shape
    model.add(Activation('relu')) #非線形回帰・勾配消失なし
    model.add(MaxPooling2D(pool_size=(2, 2))) #2*2に圧縮
    model.add(Dropout(0.25)) #drop rate:0.25

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) #平滑化：ノイズ除去

    model.add(Dense(512)) #512次元全結合
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(len(config['categories']))) #出力層(ノード数：カテゴリ数)
    model.add(Activation('softmax')) #0-1の確率に正規化

    model.compile(loss='binary_crossentropy', # crossentropy誤差
	optimizer='rmsprop', #
	metrics=['accuracy'])
    return model

# モデルを訓練する --- (※3)
def model_train(X_train, y_train):
    model = build_model()
    #データを与えて学習させる
    model.fit(X_train, y_train, batch_size=32, nb_epoch=30)
    # モデルを保存
    open(config['model_file'], 'w').write(model.to_json())
    # 学習結果を保存
    model.save_weights(config['weights_file'])
    return model

# モデルを評価する --- (※5)
def model_eval(model, X_test, y_test):
    score = model.evaluate(X_test, y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])

def main():
    data = load_data(DATA_FILE)
    model = model_train(data['X_train'], data['y_train'])
    model_eval(model, data['X_test'], data['y_test'])

### execute ###################################################################
config = load_config('./stock_keras_config.yml')
DATA_FILE = config['img_dir'] + "/stock.npy"
if __name__ == "__main__":
    main()



