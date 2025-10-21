# data_augmentation.py
import cv2
import numpy as np
from albumentations import (
    Compose, HorizontalFlip, RandomRotate90, GaussianBlur,
    RandomBrightnessContrast, ShiftScaleRotate, OpticalDistortion
)


def augment_dataset(face_images, ear_images, labels, augmentation_factor=5):
    """数据增强"""
    augmenter = Compose([
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.3),
        GaussianBlur(blur_limit=3, p=0.2),
        RandomBrightnessContrast(p=0.3),
        ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            p=0.5, border_mode=cv2.BORDER_REFLECT
        ),
        OpticalDistortion(p=0.2)
    ])

    augmented_faces = []
    augmented_ears = []
    augmented_labels = []

    # 添加原始数据
    augmented_faces.extend(face_images)
    augmented_ears.extend(ear_images)
    augmented_labels.extend(labels)

    # 生成增强数据
    for i in range(augmentation_factor):
        for face_img, ear_img, label in zip(face_images, ear_images, labels):
            augmented = augmenter(image=face_img, mask=ear_img)
            augmented_faces.append(augmented['image'])
            augmented_ears.append(augmented['mask'])
            augmented_labels.append(label)

    return np.array(augmented_faces), np.array(augmented_ears), np.array(augmented_labels)
