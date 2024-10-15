import torch
import torch.nn as nn
from tqdm import tqdm

from ChebyKAN import KAN


def test_mul():
    kan = KAN([2, 3, 3, 1], base_activation=nn.Identity())
    optimizer = torch.optim.LBFGS(kan.parameters(), lr=0.001)

    with tqdm(range(200)) as pbar:
        for i in pbar:
            loss, reg_loss = None, None

            def closure():
                optimizer.zero_grad()
                x = torch.rand(1024, 2)
                y = kan(x)
                assert y.shape == (1024, 1)
                nonlocal loss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                loss = nn.functional.mse_loss(y.squeeze(-1), u * v)
                # 由于新的 KAN 类中没有 regularization_loss 方法，可以注释掉这一行
                # reg_loss = kan.regularization_loss(1, 0)
                # (loss + 1e-5 * reg_loss).backward()
                loss.backward()
                return loss

            optimizer.step(closure)
            pbar.set_postfix(mse_loss=loss.item())
            # 如果没有 reg_loss，也可以移除或注释掉
            # pbar.set_postfix(mse_loss=loss.item(), reg_loss=reg_loss.item())

    # 如果 ChebyKANLayer 没有 spline_weight 属性，可以移除以下代码
    # for layer in kan.layers:
    #     print(layer.spline_weight)

    torch.save(kan, "model/kan_multiple_model.pth")
    torch.save(kan.state_dict(), "model/kan_multiple_weights.pth")

    # Test the trained model
    test_model(kan)


def test_model(model):
    model.eval()
    with torch.no_grad():
        test_x = torch.rand(1024, 2)
        test_y = model(test_x)
        u = test_x[:, 0]
        v = test_x[:, 1]
        expected_y = u * v
        test_loss = nn.functional.mse_loss(test_y.squeeze(-1), expected_y)
        print(f"Test Loss: {test_loss.item():.4f}")


test_mul()
