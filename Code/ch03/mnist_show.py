# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 加载数据集
# flatten=True 参数表示将28x28的图像数据展平为784维的一维数组
# normalize=True 参数表示将图像的像素值归一化到0到1之间
# one_hot_label=True 参数表示将标签转换为one-hot编码
# 训练数据和测试数据都被加载了，并且图像数据被展平为一维数组，像素值被归一化到0到1之间，标签被转换为one-hot编码
# 训练数据和测试数据的图像数据被存储在变量x_train和x_test中，标签数据被存储在变量t_train和t_test中
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)
print(x_train.shape)  # (60000, 784) x_train 是一个60000行，784列的二维数组,实际的物理意义是：60000 张 784 像素的图片
print(t_train.shape)  # (60000, 10) t_train 是一个60000行，10列的二维数组,实际的物理意义是：60000 张图片的标签
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000, 10)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
