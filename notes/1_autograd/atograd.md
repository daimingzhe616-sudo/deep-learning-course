#自动微分
深度学习框架通过自动微分来加快求导，自动微分使系统能够随后反向传播梯度。
需要一个地方来存储梯度
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
根据规定的函数计算y的值
y = 2 * torch.dot(x, x)
调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
x.grad

##非标量变量的反向传播
对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x # 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad

##分离计算
将某些计算移动到记录的计算图之外
z是y和x的函数，可以分离y来返回一个新变量u，该变量与y具有相同的值，u作为常数处理
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad == u #判断得出x的的梯度是不是被当作系数的u

##Python控制流的梯度计算
如果构建函数的计算图需要通过Python控制流，仍可以计算得到变量的梯度，在下面的函数中，while的迭代次数和if的结果都取决于a
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
计算梯度
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()