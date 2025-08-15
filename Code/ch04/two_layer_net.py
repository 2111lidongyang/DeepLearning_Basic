# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重 
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 随机初始权重
        self.params['b1'] = np.zeros(hidden_size)  # 偏置一般初始化为0
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        """
        计算损失函数。
        损失函数：用于评估模型的预测结果与真实值之间的差异。
        这里使用交叉熵误差作为损失函数。
        交叉熵误差：用于分类问题，衡量模型输出的概率分布与真实标签的差异。
        具体步骤：
        1. 计算预测结果 y
        2. 计算交叉熵误差
        3. 返回损失值
        """
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        """
        使用数值微分方法计算梯度
        输入：
        x: 输入数据
        t: 监督数据（真实标签）
        输出：
        grads: 包含每个参数的梯度的字典

        数值梯度：通过计算函数在小范围内的变化量来估计梯度。
        具体步骤：
        1. 定义损失函数关于参数的函数 loss_W
        2. 计算每个参数的数值梯度
        3. 返回梯度字典
        """
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        """
        使用反向传播算法计算梯度
        输入：
        x: 输入数据
        t: 监督数据（真实标签）
        输出：
        grads: 包含每个参数的梯度的字典

        梯度：用于更新参数，使模型向损失函数最小值的方向发展。
        具体步骤：
        1. 前向传播：计算输入数据的预测结果
        2. 反向传播：计算损失函数关于每个参数的梯度
        3. 返回梯度字典
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        """
        反向传播：计算损失函数关于每个参数的梯度
        输入：
        y: 模型的预测结果(前向传播的输出)
        t: 监督数据（真实标签）
        输出：
        梯度字典 grads，包含每个参数的梯度
        具体步骤：
        1. 计算输出层的梯度
        2. 计算隐藏层的梯度
        3. 返回梯度字典
        """
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads