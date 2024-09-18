import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Google Driveをマウントする
from google.colab import drive
drive.mount('/content/drive')

# 画像が保存されているディレクトリ
input_dir = '/content/drive/My Drive/data/data_augmentation/data'
output_dir_blur = '/content/drive/My Drive/data/data_augmentation/blur/'
output_dir_brightness = '/content/drive/My Drive/data/data_augmentation/brightness/'
output_dir_saturation = '/content/drive/My Drive/data/data_augmentation/saturation/'


# 画像処理のパラメータ
brightness_factor = 1.5  # 明るさを増加させる倍率
saturation_factor = 1.5  # 彩度を増加させる倍率
blur_kernel_size = (11, 11)  # ぼかしのカーネルサイズ
blur_sigmaX = 5  # ぼかしの標準偏差

# ディレクトリ内の全ての画像ファイルに対して処理を行う
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # jpg または png ファイルを対象にする
        # 画像のパス
        image_path = os.path.join(input_dir, filename)

        # 画像の読み込み
        image = cv2.imread(image_path)

        # 明るさの変更
        brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
        # 保存
        cv2.imwrite(os.path.join(output_dir_brightness, 'brightness_' + filename), brightened_image)

        # 彩度の変更
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:,:,1] = np.clip(hsv_image[:,:,1] * saturation_factor, 0, 255)
        saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        # 保存
        cv2.imwrite(os.path.join(output_dir_saturation, 'saturation_' + filename), saturated_image)

        # ぼかしを適用
        blurred_image = cv2.GaussianBlur(image, blur_kernel_size, sigmaX=blur_sigmaX)
        # 保存
        cv2.imwrite(os.path.join(output_dir_blur, '_' + filename), blurred_image)
