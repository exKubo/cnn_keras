from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

### 画像情報設定 ##################
convertimg_size = 100


### 学習パラメータ設定 ##################
keras_epoch = 100
keras_batch_size = 36
keras_validation_split=0.1
keras_lr = 0.0001

# グラフ出力設定
graph_dir = "graph"
loss_graph_name = "loss_graph.png"
accuracy_graph_name = "accuracy_graph.png"

# 学習用のデータを作る.
image_list = []
label_list = []

file_cnt = 0

print("Now Reading files ...")

# ./data/train 以下のorange,appleディレクトリ以下の画像を読み込む。
for dir in os.listdir("data/train"):
    if dir == ".DS_Store":
        continue

    dir1 = "data/train/" + dir 
    label = 0

    if dir == "full":    # 満タンはラベル0
        label = 0
    elif dir == "half":   # 半分はラベル1
        label = 1

    for file in os.listdir(dir1):
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(満タン:0 半分:1)
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像を100x100pixelに変換し、1要素が[R,G,B]3要素を含む配列の100x100の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath).resize((convertimg_size, convertimg_size)))
            #print(filepath)
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
            image = image.transpose(2, 0, 1)
            #print(image.shape)
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)
            file_cnt = file_cnt +1 

print("\nRead " + str(file_cnt) + " files!")

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)

# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0,0], 1 -> [0,1,0] という感じ。
Y = to_categorical(label_list)

# モデルを生成してニューラルネットを構築
### ※メモ
# Convolution2D(filter数(2の累乗), filterサイズ1(縦), filterサイズ2(横))
#
# border_mode="same"：元画像にゼロパディングを実行してフィルタリング
# (フィルター後も同じサイズ)
# 
# input_shape：最初の層での入力形式を指定
#
model = Sequential()
#model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, convertimg_size, convertimg_size)))
model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=(3, convertimg_size, convertimg_size)))
model.add(Activation("relu"))
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode=("same")))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

#model.add(Dense(200))
#model.add(Activation("relu"))
#model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))

# オプティマイザにAdamを使用
#opt = Adam(lr=0.0001)
opt = Adam(lr=keras_lr)
# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 学習を実行。10%はテストに使用
#model.fit(image_list, Y, nb_epoch=100, batch_size=36, validation_split=0.1)
hist = model.fit(image_list, Y, nb_epoch=keras_epoch, batch_size=keras_batch_size, validation_split=keras_validation_split)

# グラフプロット用データの格納
loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# lossのグラフ
plt.figure(1)
plt.plot(range(keras_epoch), loss, marker='.', label='loss')
plt.plot(range(keras_epoch), val_loss, marker='.', label='val_loss')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(os.path.join(graph_dir, loss_graph_name))

# accuracyのグラフ
plt.figure(2)
plt.plot(range(keras_epoch), acc, marker='.', label='acc')
plt.plot(range(keras_epoch), val_acc, marker='.', label='val_acc')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.savefig(os.path.join(graph_dir, accuracy_graph_name))

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
            image = np.array(Image.open(filepath).resize((convertimg_size, convertimg_size)))
            print(filepath)
            image = image.transpose(2, 0, 1)
            result = model.predict_classes(np.array([image / 255.]))
            print("label:", label, "result:", result[0])

            total += 1.

            if label == result[0]:
                ok_count += 1.

print("seikai: ", ok_count / total * 100, "%")


print('save the architecture of a model')
json_string = model.to_json()
open(os.path.join('model/','cnn_model.json'), 'w').write(json_string)
print('save weights')
model.save_weights(os.path.join('model/','cnn_model_weights.hdf5'))
