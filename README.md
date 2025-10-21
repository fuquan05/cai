基于人脸人耳的人物识别
===
本项目仅实现基于核典型相关分析（KCCA），局部二值模式纹理分析（lbp），姿态转换分析（pose）三种方法的人脸人耳识别。
使用方法如下：

1.数据准备
---
在根目录上创建文件夹（例如“dataset”），然后在文件夹里面创建子文件夹用于存放人物照片如下图：
不用在意数据的多少，少于100张时，会自动启动数据增强，以保证程序的运行。
<img width="621" height="483" alt="image" src="https://github.com/user-attachments/assets/09c38002-4263-4b3b-91c2-f57b823fcbcb" />

2.运行程序
---
修改main.py里面的数据路径，然后运行main.py。

3.运行结果
---
仅部分举例：
<img width="690" height="438" alt="0e9fd795b53df6a3fbd272bed034509e" src="https://github.com/user-attachments/assets/3ccff671-3ffa-4077-bc71-bff63ef65692" />

