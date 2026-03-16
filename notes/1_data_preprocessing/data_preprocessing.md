#数据预处理
使用pandas预处理原始数据，并将原始数据转换为张量格式

创建一个人工数据集，并存储在CSV（逗号分隔值）文件
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

从创建的CSV文件中加载原始数据集，导入pandas包并调用read_csv函数。
import pandas as pd
data = pd.read_csv(data_file)
print(data)

用插值法处理缺失的数据
通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

将“NaN”视为一个类别， 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

数值类型转换为张量格式。
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
