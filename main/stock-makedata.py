from sklearn import cross_validation
from PIL import Image
import os, glob
import numpy as np

# 分類対象のカテゴリを選ぶ --- (※1)
root_dir = "./image/"
categories = ["up_signal", "high_update", "after_up", "down_signal", "other"]
nb_classes = len(categories)
image_size = 128

# フォルダごとの画像データを読み込む --- (※2)
X = [] # 画像データ
Y = [] # ラベルデータ
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.png")
    print("---", cat, "を処理中")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB") # カラーモードの変更
        img = img.resize((image_size, image_size)) # 画像サイズの変更
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

# 学習データとテストデータを分ける --- (※3)
X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./image/stock.npy", xy)
print("ok,", len(Y))

