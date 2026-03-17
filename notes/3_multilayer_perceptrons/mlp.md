#多层感知器
初始化模型参数
784个输入特征和10个分类数据集，256个隐藏单元
每一层都要记录一个权重矩阵和一个偏置向量。
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

实现ReLU激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

实现模型
使用reshape将每个二维图像转换为一个长度为num_inputs的向量
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

损失函数
直接使用高级API中的内置函数来计算softmax和交叉熵损失
loss = nn.CrossEntropyLoss(reduction='none')

训练
与softmax回归到训练过程相同
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

评估
应用模型
d2l.predict_ch3(net, test_iter)