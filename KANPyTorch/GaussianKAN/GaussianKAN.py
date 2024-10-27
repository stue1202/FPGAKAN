import torch
import torch.nn as nn
import torch.nn.functional as F


# 定義 GaussianKANLayer 類別
class GaussianKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(GaussianKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        # 初始化高斯函數的參數：均值和標準差
        self.centers = nn.Parameter(torch.randn(num_gaussians, input_dim))
        self.stds = nn.Parameter(torch.randn(num_gaussians, input_dim).abs())
        # 初始化權重
        self.weights = nn.Parameter(torch.randn(num_gaussians, output_dim))

    def forward(self, x):
        # x 的形狀為 (batch_size, input_dim)
        batch_size = x.size(0)

        # 擴展 x 的形狀以便計算
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        centers_expanded = self.centers.unsqueeze(0)  # (1, num_gaussians, input_dim)
        stds_expanded = self.stds.unsqueeze(0)  # (1, num_gaussians, input_dim)

        # 計算高斯函數值
        diff = x_expanded - centers_expanded  # (batch_size, num_gaussians, input_dim)
        exponent = (
            -0.5 * (diff / stds_expanded) ** 2
        )  # (batch_size, num_gaussians, input_dim)
        gaussians = torch.exp(exponent.sum(dim=2))  # (batch_size, num_gaussians)

        # 計算輸出
        y = torch.matmul(gaussians, self.weights)  # (batch_size, output_dim)
        return y


# 定義 KAN 類別
class KAN(nn.Module):
    def __init__(self, layers_hidden, num_gaussians=5, base_activation=None):
        super(KAN, self).__init__()
        self.base_activation = base_activation
        # 構建 KAN 的層
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
        for layer in self.layers:
            x = layer(x)
            if self.base_activation is not None:
                x = self.base_activation(x)  # 在每一層後應用激活函數
        return x
