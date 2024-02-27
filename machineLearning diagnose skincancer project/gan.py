import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

# 데이터 폴더 경로 설정
data_dir = './datta'
class_names = ['Actinic keratosis', 'Basal cell carcinoma', 'Dermatofibroma', 'Melanoma', 'noCancer', 'Squamous cell carcinoma', 'Vascular lesion']

# DCGAN 생성자 모델 생성
def make_generator_model():
    model = models.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 3)

    return model


# 가짜 이미지 생성 함수
def generate_fake_images(generator, noise, class_name, fake_save_dir):
    class_dir = os.path.join(fake_save_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    num_images = len(glob.glob(os.path.join(data_dir, class_name, '*.jpg')))
    if num_images == 0:
        print(f"No images found for class: {class_name}")
        return

    for i in range(num_images):
        fake_image = generator(noise, training=False)
        img_name = f"fake_{i + 1}.jpg"
        img = array_to_img(fake_image[0] * 0.5 + 0.5)  # 이미지를 [-1, 1] 범위에서 [0, 1] 범위로 스케일링
        img.save(os.path.join(class_dir, img_name))

    return fake_image


# DCGAN 모델 학습 및 가짜 이미지 생성
def train_and_generate_dcgan():
    # 생성자 모델 생성
    generator = make_generator_model()

    # 생성자 컴파일
    generator.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    # DCGAN 모델 학습
    train_dcgan(generator)

    # 생성된 DCGAN 모델 로드
    generator = models.load_model('dcgan_generator.h5')

# DCGAN 모델 학습
def train_dcgan(generator):
    # 판별자 모델 생성
    discriminator = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])

    # 판별자 컴파일
    discriminator.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    # 생성자와 판별자의 입력 데이터
    random_dim = 100
    random_vector = layers.Input(shape=(random_dim,))
    generated_image = generator(random_vector)

    # 판별자 학습 중 판별 결과
    decision = discriminator(generated_image)

    # 생성자와 판별자 연결 모델 생성
    gan = models.Model(random_vector, decision)

    # 판별자 학습 중 판별 결과 비교용 레이블
    valid_labels = np.ones((64, 1))  # 실제 이미지에 대한 레이블은 1
    fake_labels = np.zeros((64, 1))  # 가짜 이미지에 대한 레이블은 0

    # 생성자와 판별자 컴파일
    gan.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    # 데이터 로드 및 전처리
    image_paths = glob.glob(os.path.join(data_dir, "*", "*.jpg"))
    num_images = len(image_paths)
    batch_size = 64

    if num_images == 0:
        print("No images found in the dataset.")
        return

    def load_image(image_path):
        img = load_img(image_path, target_size=(28, 28), color_mode='rgb')
        img = img_to_array(img) / 255.0  # 이미지를 [0, 1] 범위로 정규화
        return img

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(buffer_size=num_images)
    dataset = dataset.map(lambda x: tf.numpy_function(load_image, [x], tf.float32),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    epochs = 10

    for epoch in range(epochs):
        for images in dataset:
            # 랜덤 잡음 생성 및 가짜 이미지 생성
            noise = tf.random.normal([batch_size, random_dim])
            generated_images = generator(noise, training=True)

            # 실제 이미지와 가짜 이미지를 병합
            x = tf.concat([images, generated_images], axis=0)
            y = tf.concat([valid_labels, fake_labels], axis=0)

            # 판별자 학습
            discriminator.trainable = True
            discriminator_loss = discriminator.train_on_batch(x, y)

            # 생성자 학습
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(noise, valid_labels)

        print(f"Epoch {epoch + 1}: Discriminator Loss = {discriminator_loss}, GAN Loss = {gan_loss}")

    # 생성자 모델 저장
    generator.save('dcgan_generator.h5')


# DCGAN 모델 학습
train_and_generate_dcgan()

# 가짜 이미지 생성을 위한 랜덤 잡음 생성
random_dim = 100
noise = tf.random.normal([1, random_dim])

fake_save_dir = os.path.join(data_dir, 'fake_image')
if not os.path.exists(fake_save_dir):
    os.makedirs(fake_save_dir)

# 생성자 모델 생성
generator = make_generator_model()

for class_name in class_names:
    generate_fake_images(generator, noise, class_name, fake_save_dir)