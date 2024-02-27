import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, Input
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.decomposition import PCA
import numpy as np
train_data_dir = 'data4/Train'
test_data_dir = 'data4/Test'

# 이미지 데이터 전처리를 위한 설정


# 원본 데이터 로드 및 전처리
train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)

train_batches = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(180, 180),
        batch_size=32,
        class_mode='categorical')
test_batches = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(180, 180),
        batch_size=32,
        class_mode='categorical')

# 이미지 데이터를 1차원으로 변환
def flatten_images(generator):
    flattened_images = []
    labels = []
    for batch_images, batch_labels in generator:
        batch_flattened = batch_images.reshape(batch_images.shape[0], -1)
        flattened_images.extend(batch_flattened)
        labels.extend(batch_labels)
        if len(flattened_images) >= generator.n:
            break
    images_np = np.array(flattened_images[:generator.n])
    labels_np = np.array(labels[:generator.n])
    return images_np, labels_np

train_data, train_labels = flatten_images(train_batches)
test_data, test_labels = flatten_images(test_batches)

# PCA를 사용하여 데이터의 차원 축소
pca = PCA(n_components=128)
pca.fit(train_data)
pca_train_data = pca.transform(train_data)
pca_test_data = pca.transform(test_data)

# 분류 모델
classification_input = Input(shape=(128,))
x = Dense(256, activation='relu')(classification_input)
x = Dense(128, activation='relu')(x)
classification_output = Dense(7, activation='softmax')(x)  # 분류할 클래스의 수에 맞춰 조정하세요.
classification_model = Model(classification_input, classification_output)

# 분류 모델 컴파일 및 학습
classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
classification_model.fit(pca_train_data, train_labels, epochs=10, validation_data=(pca_test_data, test_labels))

# 모델 평가
loss, accuracy = classification_model.evaluate(pca_test_data, test_labels)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
