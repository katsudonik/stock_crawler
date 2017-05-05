from keras.utils import np_utils
import numpy as np
from keras.models import model_from_json

# 分類対象のカテゴリ
root_dir = "./image/"
categories = ["up_signal", "high_update", "after_up", "down_signal", "other"]
nb_classes = len(categories)
image_size = 128
img_height = image_size #画像の縦size(px)
img_width = image_size #画像の横size(px)
img_channels = 3 #カラー画像:3　※白黒の場合は1

def main():
    # 分類対象画像を数値化したデータをロード(result of stock-makedata.py) --- (※1)
    X_train, X_test, y_train, y_test = np.load("./image/stock.npy")
    # データを0-1に正規化する
    X_train = X_train.astype("float") / 256 #色は最大で256画素->256で割れば0-1になる
    X_test  = X_test.astype("float")  / 256
    y_train = np_utils.to_categorical(y_train, nb_classes) #transform to one-hot encoding
    y_test  = np_utils.to_categorical(y_test, nb_classes)

    # モデルを読み込む
    model = model_from_json(open('stock_model.json').read())
    # 学習結果を読み込む
    model.load_weights('./image/stock-model.hdf5')
    model.summary();
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model_eval(model, X_test, y_test)

# モデルを評価する --- (※5)
def model_eval(model, X_test, y_test):
    score = model.evaluate(X_test, y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])

if __name__ == "__main__":
    main()



