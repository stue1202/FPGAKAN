import torch
import torch.nn as nn
import torch.nn.functional as F


# 定義 ChebyKANLayer 類別
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # 由於 Chebyshev 多項式定義在 [-1, 1] 區間
        # 我們需要將 x 歸一化到 [-1, 1]，這裡使用 tanh 函數
        x = torch.tanh(x)
        # 調整 x 的形狀，並重複 degree + 1 次
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        # 對 x 應用 arccos 函數
        x = x.acos()
        # 乘以範圍為 [0, degree] 的序列
        x *= self.arange
        # 再對 x 應用 cos 函數
        x = x.cos()
        # 計算 Chebyshev 插值
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        y = y.view(-1, self.outdim)
        return y


# 定義 KAN 類別
class KAN(nn.Module):
    def __init__(self, layers_hidden, degree=3, base_activation=None):
        super(KAN, self).__init__()
        self.base_activation = base_activation
        # 構建 KAN 的層
        self.layers = self.build_layers(layers_hidden, degree)

    def build_layers(self, layers_hidden, degree):
        layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            layers.append(
                ChebyKANLayer(
                    input_dim=in_features,
                    output_dim=out_features,
                    degree=degree,
                )
            )
        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.base_activation is not None:
                x = self.base_activation(x)  # 在每一層後應用激活函數
        return x
