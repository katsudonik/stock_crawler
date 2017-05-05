from keras.utils import np_utils
import numpy as np
from keras.models import model_from_json
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

def model_eval(model, X_test, y_test):
    score = model.evaluate(X_test, y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])

def main():
    # モデルを読み込む
    model = model_from_json(open(config['model_file']).read())
    # 学習結果を読み込む
    model.load_weights(config['weights_file'])
    model.summary();
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    data = load_data(DATA_FILE)
    model_eval(model, data['X_test'], data['y_test'])

### execute ###################################################################
config = load_config('./stock_keras_config.yml')
DATA_FILE = config['img_dir'] + "/stock.npy"
if __name__ == "__main__":
    main()



