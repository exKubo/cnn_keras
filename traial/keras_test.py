from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np
from PIL import Image
import os

model_filename = 'cnn_model.json'
weights_filename = 'cnn_model_weights.hdf5'

# 学習モデルを読み込む
json_string = open(os.path.join('model/', model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join('model/',weights_filename))

label_list = []

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

for dir in os.listdir("data/test"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/test/" + dir 
    label = 0

    if dir == "full":
        label = 0
    elif dir == "half":
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            label_list.append(label)
            filepath = dir1 + "/" + file
            image = np.array(Image.open(filepath).resize((100, 100)))
            print(filepath)
            image = image.transpose(2, 0, 1)
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")