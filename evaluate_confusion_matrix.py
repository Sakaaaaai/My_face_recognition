import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.colab import drive
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Google ドライブのマウント
drive.mount('/content/drive')

# モデルのパス
model_path = '/content/drive/MyDrive/Colab Notebooks/face_recognition_model.h5'

# モデルのロード
model = tf.keras.models.load_model(model_path)

# テストデータのディレクトリ
test_dir = '/content/drive/MyDrive/data/dataset_dir/test'

# テストデータの生成器
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # 評価時はシャッフルしない
)

# モデルの評価
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Accuracy: {test_accuracy}')

# 予測結果の取得
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predicted_classes = np.round(predictions[:test_generator.samples]).astype(int).flatten()

# 実際のラベル
true_classes = test_generator.classes

# 混同行列の作成
cm = confusion_matrix(true_classes, predicted_classes)

# 正確度 (Accuracy) を計算
accuracy = accuracy_score(true_classes, predicted_classes)
print(f'Accuracy: {accuracy}')

# 適合率 (Precision) を計算
precision = precision_score(true_classes, predicted_classes)
print(f'Precision: {precision}')

# 再現率 (Recall) を計算
recall = recall_score(true_classes, predicted_classes)
print(f'Recall: {recall}')

# F1スコアを計算
f1 = f1_score(true_classes, predicted_classes)
print(f'F1 Score: {f1}')

# 混同行列のプロット
from sklearn.metrics import ConfusionMatrixDisplay

cmd = ConfusionMatrixDisplay(cm, display_labels=list(test_generator.class_indices.keys()))

# 混同行列のプロット
fig, ax = plt.subplots(figsize=(8, 8))
cmd.plot(ax=ax)
plt.title('Confusion Matrix')
plt.show()
