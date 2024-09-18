import os
import cv2
from google.colab import drive
from google.colab.patches import cv2_imshow

# Google Driveをマウント
drive.mount('/content/drive')

# 画像フォルダのパス
image_folder = '/content/drive/My Drive/data/data_resize/Before'

# 処理後の画像を保存するフォルダパス
save_folder_1 = '/content/drive/My Drive/data/data_resize/After'

# Haar Cascade分類器の読み込み（顔検出用）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# フォルダ内の画像を取得して処理
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 拡張子が.jpgまたは.pngの画像ファイルを処理する
        # 画像のパス
        image_path = os.path.join(image_folder, filename)

        # 画像の読み込み
        image = cv2.imread(image_path)

        # 画像をグレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 顔の検出
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # 検出された顔の周りを切り抜いて保存
        for i, (x, y, w, h) in enumerate(faces):
            # 顔の周囲に余裕を持たせて切り抜く
            margin_1 = 20  # 余裕のサイズ
            margin_2 = 40  # より大きな余裕

            # スライスが画像の範囲内に収まるように修正
            y1 = max(0, y - margin_1)
            y2 = min(image.shape[0], y + h + margin_1)
            x1 = max(0, x - margin_1)
            x2 = min(image.shape[1], x + w + margin_2)

            face_1 = image[y1:y2, x1:x2]

            # face_1が空でないことを確認
            if face_1.size != 0:
                # 150x150にリサイズ
                face_resized_1 = cv2.resize(face_1, (150, 150))

                # 切り抜いた顔を保存（ファイル名を変更して保存）
                save_path_1 = os.path.join(save_folder_1, f'{filename}_{i}.jpg')

                cv2.imwrite(save_path_1, face_resized_1)

                # 切り抜いた顔を表示（確認用）
                cv2_imshow(face_resized_1)
