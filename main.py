# main.py (修改后的版本)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from data_loader import BioFeatureDataLoader
from fusion_system import MultiModalFusionSystem
from data_augmentation import augment_dataset  # 导入数据增强函数

import warnings
warnings.filterwarnings(action = 'ignore')
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False

def main():
    print("=== 多模态生物特征识别系统 ===")

    # 1. 初始化数据加载器
    data_loader = BioFeatureDataLoader('dataset')  # 修改为您的数据集路径

    # 2. 加载原始数据（不进行内部增强）
    print("正在加载原始数据...")
    face_images, ear_images, labels = data_loader.load_images(augment=False)

    print(f"原始数据形状: 人脸 {face_images.shape}, 人耳 {ear_images.shape}")

    # 3. 数据增强（如果样本太少）
    if len(face_images) < 100:  # 如果样本少于100个，进行增强
        augmentation_factor = max(3, 100 // len(face_images))  # 至少增强3倍，目标100个样本
        face_images, ear_images, labels = augment_dataset(
            face_images, ear_images, labels, augmentation_factor
        )
        print(f"数据增强后形状：人脸 {face_images.shape}, 人耳 {ear_images.shape}")


    # 4. 编码标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # 5. 将图像数据展平为特征向量（用于KCCA）
    X_face_flat = face_images.reshape(len(face_images), -1)
    X_ear_flat = ear_images.reshape(len(ear_images), -1)

    print(f"展平后形状: 人脸 {X_face_flat.shape}, 人耳 {X_ear_flat.shape}")

    # 6. 特征预处理
    scaler_face = StandardScaler()
    scaler_ear = StandardScaler()

    X_face_scaled = scaler_face.fit_transform(X_face_flat)
    X_ear_scaled = scaler_ear.fit_transform(X_ear_flat)

    # 7. 划分训练测试集
    X_face_train, X_face_test, X_ear_train, X_ear_test, y_train, y_test = train_test_split(
        X_face_scaled, X_ear_scaled, y_encoded, test_size=0.3, random_state=42
    )

    print(f"训练集: 人脸 {X_face_train.shape}, 人耳 {X_ear_train.shape}")
    print(f"测试集: 人脸 {X_face_test.shape}, 人耳 {X_ear_test.shape}")

    # 8. 根据样本数量调整算法参数
    n_train_samples = X_face_train.shape[0]
    max_components = max(2, min(20, n_train_samples // 3))  # 组件数在2-20之间
    n_neighbors = max(2, min(5, n_train_samples // 4))  # 邻居数在2-5之间

    print(f"自适应参数: n_components={max_components}, n_neighbors={n_neighbors}")

    # 9. 测试三种方法
    methods = ['kcca', 'lbp', 'pose']
    results = {}

    for method in methods:
        print(f"\n=== 正在训练 {method.upper()} 方法 ===")

        try:
            fusion_system = MultiModalFusionSystem()

            # 根据方法类型传递不同的数据和参数
            if method == 'kcca':
                # 动态设置KCCA参数
                from kcca_method import KCCA_Method
                kcca_model = KCCA_Method(n_components=max_components)
                fusion_system.kcca_model = kcca_model

                # 使用预处理后的展平特征
                fusion_system.train(X_face_train, X_ear_train, y_train, method=method)
                accuracy, report = fusion_system.evaluate(X_face_test, X_ear_test, y_test, method=method)

            elif method == 'lbp':
                # 重新划分原始图像数据用于LBP
                face_train_idx, face_test_idx, ear_train_idx, ear_test_idx, y_train_lbp, y_test_lbp = train_test_split(
                    range(len(face_images)), range(len(ear_images)), y_encoded,
                    test_size=0.3, random_state=42
                )

                fusion_system.train(
                    face_images[face_train_idx],
                    ear_images[ear_train_idx],
                    y_train_lbp,
                    method=method
                )
                accuracy, report = fusion_system.evaluate(
                    face_images[face_test_idx],
                    ear_images[ear_test_idx],
                    y_test_lbp,
                    method=method
                )

            elif method == 'pose':
                # 动态设置Pose方法参数
                from pose_method import PoseTransformationMethod
                pose_model = PoseTransformationMethod(
                    n_components=max_components,
                    n_neighbors=n_neighbors
                )
                fusion_system.pose_model = pose_model

                # 重新划分原始图像数据用于姿态转换
                face_train_idx, face_test_idx, ear_train_idx, ear_test_idx, y_train_pose, y_test_pose = train_test_split(
                    range(len(face_images)), range(len(ear_images)), y_encoded,
                    test_size=0.3, random_state=42
                )

                fusion_system.train(
                    face_images[face_train_idx],
                    ear_images[ear_train_idx],
                    y_train_pose,
                    method=method
                )
                accuracy, report = fusion_system.evaluate(
                    face_images[face_test_idx],
                    ear_images[ear_test_idx],
                    y_test_pose,
                    method=method
                )

            results[method] = accuracy
            print(f"{method.upper()} 准确率: {accuracy:.4f}")
            print("分类报告:")
            print(report)

        except Exception as e:
            print(f"{method.upper()} 方法训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[method] = 0

    # 10. 可视化结果比较
    if any(results.values()):
        plt.figure(figsize=(10, 6))
        methods_names = ['KCCA', 'LBP', 'Pose Transformation']
        accuracies = [results[method] for method in methods]

        colors = ['skyblue', 'lightgreen', 'lightcoral']
        for i, (name, acc) in enumerate(zip(methods_names, accuracies)):
            plt.bar(name, acc, color=colors[i])
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom')

        plt.title('多模态生物特征识别方法性能比较')
        plt.ylabel('准确率')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n=== 最终结果比较 ===")
        for method, acc in results.items():
            status = "✓" if acc > 0 else "✗"
            print(f"{status} {method.upper()}: {acc:.4f}")
    else:
        print("所有方法都失败了，请检查数据和代码")


if __name__ == "__main__":
    main()