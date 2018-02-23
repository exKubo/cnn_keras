from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import os
import sys



img_size = 100


model_filename = 'cnn_model.json'
weights_filename = 'cnn_model_weights.hdf5'




# 学習モデルを読み込む
json_string = open(os.path.join('model/', model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join('model/',weights_filename))
demo_dir = 'demo2'

# 引数チェック
#argv = sys.argv
#argc = len(argv)
#if (argc != 2):
#    print 'Usage: python %s arg1' %argv[0]
#    quit()
#test_img = argv[1]


total = 0.
ok_count = 0.

label = 0




def img_exif(file_path):    
    # Orientation タグ値にしたがった処理
    # PIL における Rotate の角度は反時計回りが正
    convert_image = {
        1: lambda img: img,
        2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),                              # 左右反転
        3: lambda img: img.transpose(Image.ROTATE_180),                                   # 180度回転
        4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),                              # 上下反転
        5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_90),  # 左右反転＆反時計回りに90度回転
        6: lambda img: img.transpose(Image.ROTATE_270),                                   # 反時計回りに270度回転
        7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Pillow.ROTATE_270), # 左右反転＆反時計回りに270度回転
        8: lambda img: img.transpose(Image.ROTATE_90),                                    # 反時計回りに90度回転
    }
    
    img = Image.open(file_path)
    exif = img._getexif()
    orientation = exif.get(0x112, 1)
    
    new_img = convert_image[orientation](img)
    return new_img





if len(os.listdir(demo_dir)) == 0:
    print("No File!")
else:
    for file in os.listdir(demo_dir):
        if file != ".DS_Store":
            filepath = demo_dir + "/" + file
            #image = np.array(Image.open(filepath).resize((img_size, img_size)))
            img_file = img_exif(filepath)
            img_data = img_file.resize((img_size, img_size))
            img_data = img_data.crop((img_size*1/4, img_size*1/5, img_size*3/4, img_size*3/5))
            img_data.show()
            img_data = img_data.resize((img_size, img_size))
            image = np.array(img_data)
            print(filepath)
            image = image.transpose(2, 0, 1)
            result = model.predict_classes(np.array([image / 255.]))
            #print("label:", label, "result:", result[0])
            if result[0] == 0:
                result = 'Full'
            elif result[0] == 1:
                result = 'Half'
            elif result[0] == 2:
                result = 'Third'
            print("result:", result,"\n")
            #画像表示
            plt.imshow(img_file)
            plt.tick_params(labelbottom="off",bottom="off") # x軸の削除
            plt.tick_params(labelleft="off",left="off") # y軸の削除
            plt.title(result + '!', fontsize=30)
            plt.show()
    print("----------")
    print(str(len(os.listdir(demo_dir))) + " Files!")
