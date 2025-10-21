# kcca_method.py
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import rbf_kernel


class KCCA_Method:
    def __init__(self, n_components=50, kernel='rbf', gamma=0.1):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.cca = None
        self.kernel_approx_face = None
        self.kernel_approx_ear = None

    def fit(self, X_face, X_ear):
        """训练KCCA模型"""
        # 确保输入是二维数组
        if X_face.ndim > 2:
            X_face = X_face.reshape(X_face.shape[0], -1)
        if X_ear.ndim > 2:
            X_ear = X_ear.reshape(X_ear.shape[0], -1)

        n_samples = X_face.shape[0]

        # 使用Nystroem方法进行核近似
        self.kernel_approx_face = Nystroem(
            kernel=self.kernel, gamma=self.gamma,
            n_components=min(100, n_samples),
            random_state=42
        )
        self.kernel_approx_ear = Nystroem(
            kernel=self.kernel, gamma=self.gamma,
            n_components=min(100, n_samples),
            random_state=42
        )

        # 核特征映射
        X_face_kernel = self.kernel_approx_face.fit_transform(X_face)
        X_ear_kernel = self.kernel_approx_ear.fit_transform(X_ear)

        # 典型相关分析
        self.cca = CCA(n_components=self.n_components)
        self.cca.fit(X_face_kernel, X_ear_kernel)

        return self

    def transform(self, X_face, X_ear):
        """转换特征到典型相关空间"""
        # 确保输入是二维数组
        if X_face.ndim > 2:
            X_face = X_face.reshape(X_face.shape[0], -1)
        if X_ear.ndim > 2:
            X_ear = X_ear.reshape(X_ear.shape[0], -1)

        X_face_kernel = self.kernel_approx_face.transform(X_face)
        X_ear_kernel = self.kernel_approx_ear.transform(X_ear)

        X_face_c, X_ear_c = self.cca.transform(X_face_kernel, X_ear_kernel)

        # 融合特征
        fused_features = np.hstack([X_face_c, X_ear_c])
        return fused_features