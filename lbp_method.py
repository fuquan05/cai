# lbp_method.py
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.decomposition import PCA


class LBP_Method:
    def __init__(self, radius=3, n_points=24, method='uniform'):
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.pca_face = None
        self.pca_ear = None
        self.hist_bins = n_points + 2 if method == 'uniform' else 256

    def extract_lbp_features(self, images):
        """提取LBP特征"""
        lbp_features = []

        for img in images:
            # 计算LBP图像
            lbp = local_binary_pattern(img, self.n_points, self.radius, self.method)

            # 计算LBP直方图
            hist, _ = np.histogram(lbp.ravel(), bins=self.hist_bins, range=(0, self.hist_bins))

            # 归一化直方图
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-6)

            lbp_features.append(hist)

        return np.array(lbp_features)

    def extract_multiscale_lbp(self, images, radii=[1, 2, 3]):
        """提取多尺度LBP特征"""
        multiscale_features = []

        for radius in radii:
            self.radius = radius
            features = self.extract_lbp_features(images)
            multiscale_features.append(features)

        # 拼接多尺度特征
        return np.hstack(multiscale_features)

    def extract_spatial_pyramid_lbp(self, images, levels=2):
        """提取空间金字塔LBP特征"""
        pyramid_features = []

        for img in images:
            level_features = []
            h, w = img.shape

            for level in range(levels + 1):
                rows = 2 ** level
                cols = 2 ** level

                cell_height = h // rows
                cell_width = w // cols

                for i in range(rows):
                    for j in range(cols):
                        cell = img[i * cell_height:(i + 1) * cell_height,
                               j * cell_width:(j + 1) * cell_width]

                        lbp = local_binary_pattern(cell, self.n_points, self.radius, self.method)
                        hist, _ = np.histogram(lbp.ravel(), bins=self.hist_bins)
                        hist = hist.astype(np.float32)
                        hist /= (hist.sum() + 1e-6)

                        level_features.extend(hist)

            pyramid_features.append(level_features)

        return np.array(pyramid_features)