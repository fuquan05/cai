# pose_method.py
import numpy as np
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from lbp_method import LBP_Method


class PoseTransformationMethod:
    def __init__(self, n_components=50, n_neighbors=5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.pca_face = None
        self.pca_ear = None
        self.knn_model = None

    def estimate_pose(self, images):
        """估计图像姿态（简化版）"""
        # 使用PCA主成分作为姿态特征
        if self.pca_face is None:
            self.pca_face = PCA(n_components=self.n_components)
            pose_features = self.pca_face.fit_transform(images.reshape(len(images), -1))
        else:
            pose_features = self.pca_face.transform(images.reshape(len(images), -1))

        return pose_features

    def create_pose_graph(self, face_features, ear_features):
        """创建姿态图"""
        # 构建姿态邻接图
        self.knn_model = NearestNeighbors(n_neighbors=self.n_neighbors)
        combined_features = np.hstack([face_features, ear_features])
        self.knn_model.fit(combined_features)

        return self.knn_model.kneighbors_graph(combined_features)

    def pose_normalization(self, face_images, ear_images, target_pose=None):
        """姿态归一化"""
        # 提取姿态特征
        face_pose_features = self.estimate_pose(face_images)
        ear_pose_features = self.estimate_pose(ear_images)

        if target_pose is None:
            # 使用平均姿态作为目标姿态
            target_pose_face = np.mean(face_pose_features, axis=0)
            target_pose_ear = np.mean(ear_pose_features, axis=0)
        else:
            target_pose_face, target_pose_ear = target_pose

        # 简单的线性姿态转换
        normalized_faces = self.linear_pose_transform(face_images, face_pose_features, target_pose_face)
        normalized_ears = self.linear_pose_transform(ear_images, ear_pose_features, target_pose_ear)

        return normalized_faces, normalized_ears

    def linear_pose_transform(self, images, pose_features, target_pose):
        """线性姿态转换"""
        normalized_images = []

        for i, img in enumerate(images):
            pose_diff = target_pose - pose_features[i]
            # 简化的线性变换（实际应用中应使用更复杂的方法）
            transform_matrix = np.eye(3)
            transform_matrix[0, 2] = pose_diff[0] * 0.1 if len(pose_diff) > 0 else 0
            transform_matrix[1, 2] = pose_diff[1] * 0.1 if len(pose_diff) > 1 else 0

            rows, cols = img.shape
            transformed = cv2.warpAffine(img, transform_matrix[:2], (cols, rows))
            normalized_images.append(transformed)

        return np.array(normalized_images)

    def extract_pose_invariant_features(self, face_images, ear_images):
        """提取姿态不变特征"""
        # 姿态归一化
        normalized_faces, normalized_ears = self.pose_normalization(face_images, ear_images)

        # 使用LBP提取纹理特征（可以与LBP方法结合）
        lbp_extractor = LBP_Method()
        face_features = lbp_extractor.extract_spatial_pyramid_lbp(normalized_faces)
        ear_features = lbp_extractor.extract_spatial_pyramid_lbp(normalized_ears)

        return face_features, ear_features