import torch

batch_size = 8
in_features = 5
grid_size = 10
spline_order = 3

# 模擬輸入
x = torch.randn(batch_size, in_features)
grid = torch.randn(in_features, grid_size + 2 * spline_order + 1)

# 計算 B-Spline bases
b_spline_bases = ((x.unsqueeze(-1) >= grid[:, :-1]) & (x.unsqueeze(-1) < grid[:, 1:])).to(torch.float)

for k in range(1, spline_order + 1):
    b_spline_bases = (
        (x.unsqueeze(-1) - grid[:, :-(k+1)]) / (grid[:, k:-1] - grid[:, :-(k+1)]) * b_spline_bases[:, :, :-1]
    ) + (
        (grid[:, k+1:] - x.unsqueeze(-1)) / (grid[:, k+1:] - grid[:, 1:-k]) * b_spline_bases[:, :, 1:]
    )
    print("Final Shape of bases:", b_spline_bases.shape)
