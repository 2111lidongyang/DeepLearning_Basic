# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    # 获取数据集
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test  # 返回测试集和测试标签


def init_network():
    # 初始化网络
    # sample_weight.pkl本质就是一个字典，字典的键是层的名称，值是层的参数
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """
    对于预测，实际上就是进行一个前向传播，最终得到预测结果。
    前向传播：就是从输入层开始，一层一层地计算，最终得到输出层的结果。
    """
    # 获取网络参数和偏置
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    

    a1 = np.dot(x, w1) + b1  # 计算第一层的加权和
    z1 = sigmoid(a1)  # 使用激活函数 sigmoid 进行非线性转换
    a2 = np.dot(z1, w2) + b2  # 计算第二层的加权和
    z2 = sigmoid(a2)  # 使用激活函数 sigmoid 进行非线性转换
    a3 = np.dot(z2, w3) + b3  # 计算第三层的加权和
    y = softmax(a3)  # 使用 softmax 函数计算输出层的结果
    return y


x, t = get_data()
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
