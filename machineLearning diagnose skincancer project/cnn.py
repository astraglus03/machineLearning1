import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential


# 이미지 로드 및 전처리 함수
def load_images(path, target_size):
    images = []
    for img_file in os.listdir(path):
        img = load_img(os.path.join(path, img_file), target_size=target_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)


# 이미지 경로
data_dir = 'data4'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
image_categories = os.listdir(train_dir)
num_categories = len(image_categories)

# 이미지 로드
target_size = (180, 180)
train_images = []
test_images = []

for category in image_categories:
    category_train_dir = os.path.join(train_dir, category)
    category_test_dir = os.path.join(test_dir, category)
    train_images.extend(load_images(category_train_dir, target_size))
    test_images.extend(load_images(category_test_dir, target_size))

train_images = np.array(train_images)
test_images = np.array(test_images)

# 이미지 정규화 (0과 1 사이로 스케일 조정)
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 이미지 분류를 위한 레이블 생성
train_labels = []
test_labels = []

for i, category in enumerate(image_categories):
    category_train_dir = os.path.join(train_dir, category)
    category_test_dir = os.path.join(test_dir, category)

    train_images_category = load_images(category_train_dir, target_size)
    test_images_category = load_images(category_test_dir, target_size)

    train_labels.extend([i] * len(train_images_category))
    test_labels.extend([i] * len(test_images_category))

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

classification_model = Sequential()
classification_model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(180, 180, 3)))
classification_model.add(MaxPooling2D(pool_size=(2, 2)))
classification_model.add(Conv2D(32, (3, 3), activation='relu'))
classification_model.add(MaxPooling2D(pool_size=(2, 2)))
classification_model.add(Conv2D(64, (3, 3), activation='relu'))
classification_model.add(MaxPooling2D(pool_size=(2, 2)))
classification_model.add(Flatten())
classification_model.add(Dense(128, activation='relu'))
classification_model.add(Dense(7, activation='softmax'))

# 모델 컴파일 및 학습
classification_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classification_model.fit(train_images, train_labels,
                                                  epochs=20,
                                                  batch_size=32,
                                                  shuffle=True,
                                                  validation_data=(test_images, test_labels))

classification_model.save('skin_model.h5')

# acc = classification_model.history['accuracy']
# val_acc = classification_model.history['val_accuracy']
# loss = classification_model.history['loss']
# val_loss = classification_model.history['val_loss']
# epochs_range = range(20)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# 이미지 시각화
n = 10  # 시각화할 이미지 개수
plt.figure(figsize=(20, 6))
for i in range(n):
    # 분류 결과
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(test_images[i])
    predicted_label = np.argmax(classification_model.predict(np.expand_dims(test_images[i], axis=0)))
    plt.title(image_categories[predicted_label])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()