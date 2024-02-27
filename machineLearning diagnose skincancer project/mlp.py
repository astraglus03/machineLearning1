import os
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터셋 경로 설정
dataset_path = 'data4/Train'  # 데이터셋 폴더 경로
class_labels = sorted(os.listdir(dataset_path))  # 피부암 종류 폴더의 이름을 정렬하여 클래스 레이블로 사용

# 데이터셋 로딩 및 전처리
X = []  # 이미지 데이터를 저장할 리스트
y = []  # 클래스 레이블을 저장할 리스트

for label, class_name in enumerate(class_labels):
    class_path = os.path.join(dataset_path, class_name)
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        image = Image.open(file_path)
        image = image.resize((64, 64))  # 이미지 크기 조정 (원하는 크기로 변경 가능)
        image = np.array(image)  # 이미지를 배열로 변환
        X.append(image.flatten())  # 이미지를 1차원 벡터로 펼쳐서 저장
        y.append(label)  # 해당 이미지의 클래스 레이블 저장

X = np.array(X)  # 이미지 데이터 배열로 변환
y = np.array(y)  # 클래스 레이블 배열로 변환

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP 분류기 정의 및 학습
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)  # 은닉층 크기와 최대 반복 횟수 설정
mlp.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = mlp.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
