import matplotlib.pyplot as plt
import cv2
import os

# 誤分類された画像のインデックスを取得
misclassified_indices = np.where(predicted_classes != true_classes)[0]

# 誤分類された画像を表示する関数
def display_misclassified_images(test_generator, misclassified_indices):
    plt.figure(figsize=(20, 20))
    num_images = len(misclassified_indices)
    for i, index in enumerate(misclassified_indices):
        # 画像のパスを取得
        img_path = test_generator.filepaths[index]
        # 画像のファイル名を取得
        img_name = os.path.basename(img_path)
        # 画像を読み込む
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # サブプロットを作成
        plt.subplot(num_images // 5 + 1, 5, i + 1)
        plt.imshow(img)
        plt.title(f'Name: {img_name}\nTrue: {true_classes[index]}, Pred: {predicted_classes[index]}', fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 全ての誤分類された画像を表示
display_misclassified_images(test_generator, misclassified_indices)
