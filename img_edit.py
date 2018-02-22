# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import os


# ヒストグラム均一化
def equalizeHistRGB(src):
    
    RGB = cv2.split(src)
    Blue   = RGB[0]
    Green = RGB[1]
    Red    = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])

    img_hist = cv2.merge([RGB[0],RGB[1], RGB[2]])
    return img_hist

# ガウシアンノイズ
def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
    sigma = 15 # ※
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss
    
    return noisy

# salt&pepperノイズ
def addSaltPepperNoise(src):
    row,col,ch = src.shape
    s_vs_p = 0.5
    amount = 0.004 # ※
    out = src.copy()
    # Salt mode
    num_salt = np.ceil(amount * src.size * s_vs_p)
    coords = [np.random.randint(0, i-1 , int(num_salt))
                 for i in src.shape]
    out[coords[:-1]] = (255,255,255)

    # Pepper mode
    num_pepper = np.ceil(amount* src.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1 , int(num_pepper))
             for i in src.shape]
    out[coords[:-1]] = (0,0,0)
    return out

#if __name__ == '__main__':
def img_edit(file_path,):
    # ルックアップテーブルの生成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table
    gamma1 = 0.75 # ※
    gamma2 = 1.5 # ※

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )
    LUT_G1 = np.arange(256, dtype = 'uint8' )
    LUT_G2 = np.arange(256, dtype = 'uint8' )

    LUTs = []

    # 平滑化用
    average_square = (10,10) # ※

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
               
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
                                  
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # その他LUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1) 
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

    LUTs.append(LUT_HC)
    LUTs.append(LUT_LC)
    LUTs.append(LUT_G1)
    LUTs.append(LUT_G2)

    # 画像の読み込み
    #img_src = cv2.imread(sys.argv[1], 1)
    img_src = cv2.imread(file_path, 1)
    trans_img = []
    trans_img.append(img_src)
    
    # LUT変換
    for i, LUT in enumerate(LUTs):
        trans_img.append( cv2.LUT(img_src, LUT))

    # 平滑化      
    trans_img.append(cv2.blur(img_src, average_square))      

    # ヒストグラム均一化
    trans_img.append(equalizeHistRGB(img_src))

    # ノイズ付加
    trans_img.append(addGaussianNoise(img_src))
    trans_img.append(addSaltPepperNoise(img_src))

    # 反転
    # flip_img = []
    # for img in trans_img:
    #     flip_img.append(cv2.flip(img, 1))
    # trans_img.extend(flip_img)
    #trans_img.append(cv2.flip(img_src, 1))


    # 保存
    if not os.path.exists("edit_output"):
        os.mkdir("edit_output")

    if not os.path.exists("edit_output_compare"):
        os.mkdir("edit_output_compare")

    #base =  os.path.splitext(os.path.basename(sys.argv[1]))[0] + "_"
    base =  os.path.splitext(os.path.basename(file_path))[0] + "_"
    img_src.astype(np.float64)
    for i, img in enumerate(trans_img):
        edit_flg=0
        ### 実行設定 #####
        edit_original_flg = 1
        edit_highContrast_flg = 1
        edit_lowContrast_flg = 0
        edit_gammaConvert1_flg = 1
        edit_gammaConvert2_flg = 0
        edit_smoothing_flg = 1
        edit_histogram_flg = 0
        edit_gaussianNoise_flg = 0
        edit_saltPepperNoise_flg = 0
        edit_inversion_flg = 0
        ###################
        
        # ラベル作成
        if i == 0 and edit_original_flg == 1:
            editing = 'original'
            edit_flg = 1
        elif i == 1 and edit_highContrast_flg == 1:
            editing = 'highContrast'
            edit_flg = 1
        elif i == 2 and edit_lowContrast_flg == 1:
            editing = 'lowContrast'
            edit_flg = 1
        elif i == 3 and edit_gammaConvert1_flg == 1:
            editing = 'gammaConvert1'
            edit_flg = 1
        elif i == 4 and edit_gammaConvert2_flg == 1:
            editing = 'gammaConvert2'
            edit_flg = 1
        elif i == 5 and edit_smoothing_flg == 1:
            editing = 'smoothing'
            edit_flg = 1
        elif i == 6 and edit_histogram_flg == 1:
            editing = 'histogram'
            edit_flg = 1
        elif i == 7 and edit_gaussianNoise_flg == 1:
            editing = 'gaussianNoise'
            edit_flg = 1
        elif i == 8 and edit_saltPepperNoise_flg == 1:
            editing = 'saltPepperNoise'
            edit_flg = 1
        elif i == 9 and edit_inversion_flg == 1:
            editing = 'inversion'
            edit_flg = 1
        else:
            editing = 'err'
        
        # 保存
        if edit_flg == 1:
            cv2.imwrite("edit_output/" + base + editing + ".jpg" ,img) 
            # 比較用画像
            #cv2.imwrite("edit_output_compare/" + base + editing + ".jpg" ,cv2.hconcat([img_src.astype(np.float64), img.astype(np.float64)]))
            

for file in os.listdir("edit_input"):
    os.path.join('edit_input', file)
    img_edit(os.path.join('edit_input', file))