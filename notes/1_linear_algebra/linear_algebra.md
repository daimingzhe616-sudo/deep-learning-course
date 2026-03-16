#线性代数
标量由只有一个元素的张量表示
import torch
x = torch.tensor(3.0)
y = torch.tensor(2.0)
x + y, x * y, x / y, x**y

向量可以被视为标量值组成的列表
x = torch.arange(4)
通过张量的索引来访问任一元素
x[3]

调用Python的内置len()函数来访问张量的长度
len(x)

通过指定两个分量$m$和$n$来创建一个形状为$m \times n$的矩阵
A = torch.arange(20).reshape(5, 4)

矩阵的转置
A.T

对称矩阵等于其转置
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B == B.T
输出
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])

处理图像时，张量将变得更加重要，图像以$n$维数组形式出现
X = torch.arange(24).reshape(2, 3, 4)

给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B

两个矩阵的按元素乘法称为Hadamard积
A * B

将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape

##降维
计算其元素的和，可以降低张量的维度
x = torch.arange(4, dtype=torch.float32)
x, x.sum()

指定张量沿哪一个轴来通过求和降低维度（轴0）
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape

指定axis=1将通过汇总所有列的元素降维（轴1）
A_sum_axis1 = A.sum(axis=1)

沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和。
A.sum(axis=[0, 1])  # 结果和A.sum()相同

计算平均值的函数也可以沿指定轴降低张量的维度
A.mean(axis=0), A.sum(axis=0) / A.shape[0]

非降维求和
sum_A = A.sum(axis=1, keepdims=True)

某个轴计算A元素的累积总和，不会沿任何轴降低输入张量的维度
A.cumsum(axis=0)

点积，相同位置的按元素乘积的和
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
可以通过执行按元素乘法，然后进行求和来表示两个向量的点积
torch.sum(x * y)

使用张量表示矩阵-向量积，A的列维数（沿轴1的长度）必须与x的维数（其长度）相同。
torch.mv(A, x)
在A和B上执行矩阵乘法。 这里的A是一个5行4列的矩阵，B是一个4行3列的矩阵。 两者相乘后，我们得到了一个5行3列的矩阵。
torch.mm(A, B)

$L_2$范数是向量元素平方和的平方根，在$L_2$范数中常常省略下标$2$
u = torch.tensor([3.0, -4.0])
torch.norm(u)
输出
tensor(5.)
$L_1$范数，它表示为向量元素的绝对值之和
torch.abs(u).sum()
输出
tensor(7.)
Frobenius范数（Frobenius norm）是矩阵元素平方和的平方根
torch.norm(torch.ones((4, 9)))