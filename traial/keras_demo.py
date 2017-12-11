from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np
from PIL import Image
import os
import sys

model_filename = 'cnn_model.json'
weights_filename = 'cnn_model_weights.hdf5'

# 学習モデルを読み込む
json_string = open(os.path.join('model/', model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join('model/',weights_filename))
demo_dir = 'demo'

# 引数チェック
#argv = sys.argv
#argc = len(argv)
#if (argc != 2):
#    print 'Usage: python %s arg1' %argv[0]
#    quit()
#test_img = argv[1]

# テスト用ディレクトリ(./data/train/)の画像でチェック。正解率を表示する。
total = 0.
ok_count = 0.

label = 0

for file in os.listdir(demo_dir):
    if file != ".DS_Store":
        filepath = demo_dir + "/" + file
        image = np.array(Image.open(filepath).resize((100, 100)))
        print(filepath)
        image = image.transpose(2, 0, 1)
        result = model.predict_classes(np.array([image / 255.]))
        #print("label:", label, "result:", result[0])
        if result[0] == 0:
            result = 'full'
        elif result[0] == 1:
            result = 'half'
        print("result:", result,"\n")

