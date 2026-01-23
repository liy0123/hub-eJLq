import torch
import torch.nn as nn
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
# 角度矩阵
X_numpy = np.random.rand(200, 1) * 2 * np.pi  # 200行 1列的的矩阵进行2π的乘

# 根据角度矩阵生成 sin函数对应的数据 同时加入干涉
y_numpy = np.sin(X_numpy) + np.random.rand(200, 1) * 0.1

# X矩阵转为 tensor
X = torch.from_numpy(X_numpy).float()  # torch 中 所有的计算 通过tensor 计算
# Y矩阵转为 tensor
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 直接创建参数张量 a 和 b
# torch.randn() 生成随机值作为初始值。
# y = a * x + b
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
# a = torch.randn(1, requires_grad=True, dtype=torch.float)
# b = torch.randn(1, requires_grad=True, dtype=torch.float)

print("---" * 10)


# 创建 sin函数的网络模型 3层

class sinFitModel(nn.Module):
    def __init__(self, input_dim=1, hidden=16, output_dim=1):
        super(sinFitModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # sin() 函数更加适合周期函数

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss()  # 回归任务
# a * x + b 《 - 》  y'

# 优化器现在直接传入我们手动创建的参数 [a, b]。
# PyTorch 会自动根据这些参数的梯度来更新它们。
# 初始化 网络模型
model = sinFitModel(1, 16, 1)
# 使用Adam 非线性优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 5000
loss_latest = None
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数
    loss_latest = loss.item()
    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print(f"拟合的损失 b: {loss_latest:.4f}")
print("---" * 10)

# 生成等间距的 X矩阵 200行1列
X_test_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
# 生成X矩阵的tensor
X_test = torch.from_numpy(X_test_numpy).float()

# 6. 绘制结果

# 模型评估
model.eval()
with torch.no_grad():
    # 根据模拟数据生成sin 函数
    y_predicted = model(X_test).numpy()
# 真实的正弦函数
y_true = np.sin(X_numpy)

# 生成画布 1000*600px的
plt.figure(figsize=(10, 6))
#  生成 点位的
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
# 生成曲线
plt.plot(X_test_numpy, y_predicted, label=f'Model: y = sin(x)', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
# 网格
plt.grid(True)
# 留存图片
plt.savefig("sinFitModel.png")
# 显示图片
plt.show()
