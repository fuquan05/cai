# data_loader.py
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import albumentations as A


class BioFeatureDataLoader:
    def __init__(self, data_path, img_size=(128, 128)):
        self.data_path = data_path
        self.img_size = img_size
        self.face_images = []
        self.ear_images = []
        self.labels = []
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def load_images(self, augment=True):
        """加载人脸和人耳图像对"""
        # 假设数据组织格式：data_path/person_id/face.jpg, ear.jpg
        for person_id in os.listdir(self.data_path):
            person_path = os.path.join(self.data_path, person_id)

            if os.path.isdir(person_path):
                face_path = os.path.join(person_path, "face.jpg")
                ear_path = os.path.join(person_path, "ear.jpg")

                if os.path.exists(face_path) and os.path.exists(ear_path):
                    # 加载人脸图像
                    face_img = cv2.imread(face_path)
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_img = cv2.resize(face_img, self.img_size)

                    # 加载人耳图像
                    ear_img = cv2.imread(ear_path)
                    ear_img = cv2.cvtColor(ear_img, cv2.COLOR_BGR2GRAY)
                    ear_img = cv2.resize(ear_img, self.img_size)

                    self.face_images.append(face_img)
                    self.ear_images.append(ear_img)
                    self.labels.append(person_id)

                    if augment:
                        # 数据增强
                        augmented = self.transform(image=face_img, mask=ear_img)
                        self.face_images.append(augmented['image'])
                        self.ear_images.append(augmented['mask'])
                        self.labels.append(person_id)

        return np.array(self.face_images), np.array(self.ear_images), np.array(self.labels)

    def preprocess_features(self, face_features, ear_features):
        """特征预处理"""
        scaler_face = StandardScaler()
        scaler_ear = StandardScaler()

        face_features_scaled = scaler_face.fit_transform(face_features)
        ear_features_scaled = scaler_ear.fit_transform(ear_features)

        return face_features_scaled, ear_features_scaled