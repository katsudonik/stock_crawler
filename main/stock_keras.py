from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np

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
    # モデルを訓練し評価する
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

# モデルを構築 --- (※2)
def build_model():
    model = Sequential()
    #keras2->Conv2D
    model.add(Conv2D(32, (3, 3),#filter(convolution window/kernel):3*3size, 32 sheet
    padding='same',
	input_shape=(img_height, img_width, img_channels))) #channels_last #first layer in a model, provide input_shape
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) #2*2に圧縮
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten()) 
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])
    return model

# モデルを訓練する --- (※3)
def model_train(X_train, y_train):
    model = build_model()
    #データを与えて学習させる
    model.fit(X_train, y_train, batch_size=32, nb_epoch=30)
    # 学習したモデルを保存する --- (※4)->保存したモデルは再利用できる
    # モデルを保存
    model_json_str = model.to_json()
    open('stock_model.json', 'w').write(model_json_str)
    # 学習結果を保存
    hdf5_file = "./image/stock-model.hdf5"
    model.save_weights(hdf5_file)
    return model

# モデルを評価する --- (※5)
def model_eval(model, X_test, y_test):
    score = model.evaluate(X_test, y_test)
    print('loss=', score[0])
    print('accuracy=', score[1])

if __name__ == "__main__":
    main()



