import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定义 GaussianKANLayer 类
class GaussianKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(GaussianKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        # 初始化参数
        self.centers = nn.Parameter(torch.randn(num_gaussians, input_dim) * 0.1)
        self.log_stds = nn.Parameter(torch.zeros(num_gaussians, input_dim))  # 使用对数标准差
        self.weights = nn.Parameter(torch.randn(num_gaussians, output_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_gaussians, input_dim)
        stds_expanded = torch.exp(self.log_stds).unsqueeze(0) + 1e-6  # 确保标准差为正

        diff = x_expanded - centers_expanded  # (batch_size, num_gaussians, input_dim)
        exponent = -0.5 * (diff / stds_expanded) ** 2
        exponent = exponent.sum(dim=2)  # (batch_size, num_gaussians)
        exponent = torch.clamp(exponent, min=-50, max=50)  # 防止数值不稳定
        gaussians = torch.exp(exponent)

        y = torch.matmul(gaussians, self.weights) + self.bias  # (batch_size, output_dim)
        return y

# 定义 KAN 类
class KAN(nn.Module):
    def __init__(self, layers_hidden, num_gaussians=20, base_activation=None):
        super(KAN, self).__init__()
        self.base_activation = base_activation
        self.layers = self.build_layers(layers_hidden, num_gaussians)

    def build_layers(self, layers_hidden, num_gaussians):
        layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            layers.append(
                GaussianKANLayer(
                    input_dim=in_features,
                    output_dim=out_features,
                    num_gaussians=num_gaussians,
                )
            )
        return layers

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            # 仅在非最后一层使用激活函数
            if self.base_activation is not None and idx < len(self.layers) - 1:
                x = self.base_activation(x)
        return x

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
batch_size = 64
epochs = 10
learning_rate = 0.01  # 调整学习率

# MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transform, download=True)

# 创建小数据集进行过拟合测试
small_train_dataset = torch.utils.data.Subset(train_dataset, list(range(500)))
train_loader = torch.utils.data.DataLoader(dataset=small_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 初始化模型
input_dim = 28 * 28  # MNIST 图片大小为 28x28
output_dim = 10      # MNIST 有 10 个类别
layers_hidden = [input_dim, 128, 64, output_dim]
num_gaussians = 50   # 增加高斯数量
base_activation = None  # 先不使用激活函数

model = KAN(layers_hidden=layers_hidden, num_gaussians=num_gaussians,
            base_activation=base_activation)
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28).to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()

        # 检查梯度
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / 5:.4f}, Gradient Norm: {total_norm:.6f}')
            running_loss = 0.0

    # 在训练集上评估（由于使用的是小数据集）
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in train_loader:
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'Epoch [{epoch + 1}/{epochs}], Training Accuracy: {100 * correct / total:.2f}%')
