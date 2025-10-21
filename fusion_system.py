# fusion_system.py
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


class MultiModalFusionSystem:
    def __init__(self, fusion_method='feature_level'):
        self.fusion_method = fusion_method
        self.classifier = None
        self.kcca_model = None
        self.lbp_extractor = None
        self.pose_model = None

    def feature_level_fusion(self, face_features, ear_features):
        """特征级融合"""
        return np.hstack([face_features, ear_features])

    def train(self, X_train_face, X_train_ear, y_train, method='kcca'):
        """训练融合系统"""
        if method == 'kcca':
            from kcca_method import KCCA_Method
            self.kcca_model = KCCA_Method()

            # 确保数据是二维的
            if X_train_face.ndim > 2:
                X_train_face = X_train_face.reshape(len(X_train_face), -1)
            if X_train_ear.ndim > 2:
                X_train_ear = X_train_ear.reshape(len(X_train_ear), -1)

            fused_features = self.kcca_model.fit(X_train_face, X_train_ear) \
                .transform(X_train_face, X_train_ear)

        elif method == 'lbp':
            from lbp_method import LBP_Method
            self.lbp_extractor = LBP_Method()

            # 确保数据是图像格式
            if X_train_face.ndim == 2:
                # 如果是展平的数据，需要重新reshape为图像
                img_size = int(np.sqrt(X_train_face.shape[1]))
                X_train_face = X_train_face.reshape(-1, img_size, img_size)
                X_train_ear = X_train_ear.reshape(-1, img_size, img_size)

            face_features = self.lbp_extractor.extract_spatial_pyramid_lbp(X_train_face)
            ear_features = self.lbp_extractor.extract_spatial_pyramid_lbp(X_train_ear)
            fused_features = self.feature_level_fusion(face_features, ear_features)

        elif method == 'pose':
            from pose_method import PoseTransformationMethod
            self.pose_model = PoseTransformationMethod()

            if X_train_face.ndim == 2:
                img_size = int(np.sqrt(X_train_face.shape[1]))
                X_train_face = X_train_face.reshape(-1, img_size, img_size)
                X_train_ear = X_train_ear.reshape(-1, img_size, img_size)

            face_features, ear_features = self.pose_model.extract_pose_invariant_features(
                X_train_face, X_train_ear)
            fused_features = self.feature_level_fusion(face_features, ear_features)

        else:
            raise ValueError(f"不支持的方法: {method}")

        # 训练分类器
        self.classifier = SVC(kernel='rbf', probability=True, random_state=42)
        self.classifier.fit(fused_features, y_train)

        return self

    def predict(self, X_face, X_ear, method):
        """预测身份"""
        if method == 'kcca' and self.kcca_model:
            # 确保数据是二维的
            if X_face.ndim > 2:
                X_face = X_face.reshape(len(X_face), -1)
            if X_ear.ndim > 2:
                X_ear = X_ear.reshape(len(X_ear), -1)

            features = self.kcca_model.transform(X_face, X_ear)

        elif method == 'lbp' and self.lbp_extractor:
            # 确保数据是图像格式
            if X_face.ndim == 2:
                img_size = int(np.sqrt(X_face.shape[1]))
                X_face = X_face.reshape(-1, img_size, img_size)
                X_ear = X_ear.reshape(-1, img_size, img_size)

            face_features = self.lbp_extractor.extract_spatial_pyramid_lbp(X_face)
            ear_features = self.lbp_extractor.extract_spatial_pyramid_lbp(X_ear)
            features = self.feature_level_fusion(face_features, ear_features)

        elif method == 'pose' and self.pose_model:
            if X_face.ndim == 2:
                img_size = int(np.sqrt(X_face.shape[1]))
                X_face = X_face.reshape(-1, img_size, img_size)
                X_ear = X_ear.reshape(-1, img_size, img_size)

            face_features, ear_features = self.pose_model.extract_pose_invariant_features(
                X_face, X_ear)
            features = self.feature_level_fusion(face_features, ear_features)

        else:
            raise ValueError("Method not trained properly")

        return self.classifier.predict(features)

    def evaluate(self, X_face, X_ear, y_true, method):
        """评估系统性能"""
        y_pred = self.predict(X_face, X_ear, method)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        return accuracy, report