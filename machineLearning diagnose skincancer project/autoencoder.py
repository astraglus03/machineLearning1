import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import matplotlib.pyplot as plt


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
target_size = (224, 224)
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

# 오토인코더 모델 정의
input_shape = train_images[0].shape
input_img = Input(shape=input_shape)

# 인코더 부분
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# 디코더 부분
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# 오토인코더 모델 생성
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

# from tensorflow.keras.callbacks import EarlyStopping
#
# # EarlyStopping 콜백 정의
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 오토인코더 모델 훈련
history = autoencoder.fit(train_images, train_images,
                          epochs=5,
                          batch_size=1,
                          shuffle=True,
                          validation_data=(test_images, test_images),  # callbacks=[early_stopping]
                          )

# 분류 모델 정의
flatten = Flatten()(encoded)
dense = Dense(64, activation='relu')(flatten)
output = Dense(num_categories, activation='softmax')(dense)

classification_model = Model(input_img, output)
classification_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                             loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 분류 모델 훈련
classification_history = classification_model.fit(train_images, train_labels,
                                                  epochs=20,
                                                  batch_size=64,
                                                  shuffle=True,
                                                  validation_data=(test_images, test_labels),
                                                  # callbacks=[early_stopping]
                                                  )

# 재구성된 이미지 예측
reconstructed_images = autoencoder.predict(test_images)

# 이미지 시각화
n = 10  # 시각화할 이미지 개수
plt.figure(figsize=(20, 6))
for i in range(n):
    # 원본 이미지
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_images[i])
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 이미지
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed_images[i])
    plt.title("Reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 분류 결과
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(test_images[i])
    predicted_label = np.argmax(classification_model.predict(np.expand_dims(test_images[i], axis=0)))
    plt.title(f"Predicted: {image_categories[predicted_label]}")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()