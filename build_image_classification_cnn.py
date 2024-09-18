import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from google.colab import drive
import os

# Google ドライブのマウント
drive.mount('/content/drive')

# データセットのディレクトリ
train_dir = '/content/drive/MyDrive/data/dataset_dir/train'
validation_dir = '/content/drive/MyDrive/data/dataset_dir/validation'

# ディレクトリが存在するか確認
if os.path.exists(train_dir) and os.path.exists(validation_dir):
    print("Both train and validation directories exist")
else:
    print("Train or validation directory does not exist")

# データ拡張の設定
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# 訓練データの生成器
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # 自分自身 vs 他の人物の2クラス分類
)

# 検証データの生成器
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# モデルの構築
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,  # ここを変更
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# モデルの保存
model.save('/content/drive/MyDrive/Colab Notebooks/face_recognition_model.h5')

# 訓練結果の可視化
plt.figure(figsize=(12, 4))

# 訓練および検証の損失の履歴をプロット
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 訓練および検証の正答率の履歴をプロット
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
