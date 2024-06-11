import os
import numpy as np
import torch
import torch.nn as nn
from EfficientKAN import KAN

# 加載已訓練的模型
model_path = 'model/kan_multiple_weights.pth'
model = KAN([2, 3, 3, 1], base_activation=nn.Identity)
model.load_state_dict(torch.load(model_path))

# 量化工具
def quantize_tensor(tensor, num_bits):
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    quantized_tensor = (tensor / scale + zero_point).round().clamp(qmin, qmax)
    quantized_tensor = quantized_tensor.int()

    return quantized_tensor, scale, zero_point

# 將浮點數轉換為二進制字符串
def float_to_binary(value):
    import struct
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return f'{d:064b}'

# 將整數轉換為二進制字符串
def int_to_binary(value, num_bits):
    return f'{value:0{num_bits}b}'

# 將權重保存到單獨的 TXT 檔案
def save_layer_weights_to_txt(model, layer_index, base_dir, num_bits):
    layer = model.layers[layer_index]
    folder_name = os.path.join(base_dir, f"{num_bits}bits")
    os.makedirs(folder_name, exist_ok=True)

    base_weight_file = os.path.join(folder_name, f"layer{layer_index}_base_weight.txt")
    spline_weight_file = os.path.join(folder_name, f"layer{layer_index}_spline_weight.txt")
    spline_scaler_file = os.path.join(folder_name, f"layer{layer_index}_spline_scaler.txt")

    base_weight_scale_file = os.path.join(folder_name, f"layer{layer_index}_scale_base_weight.txt")
    base_weight_zero_point_file = os.path.join(folder_name, f"layer{layer_index}_zero_point_base_weight.txt")

    spline_weight_scale_file = os.path.join(folder_name, f"layer{layer_index}_scale_spline_weight.txt")
    spline_weight_zero_point_file = os.path.join(folder_name, f"layer{layer_index}_zero_point_spline_weight.txt")

    spline_scaler_scale_file = os.path.join(folder_name, f"layer{layer_index}_scale_spline_scaler.txt")
    spline_scaler_zero_point_file = os.path.join(folder_name, f"layer{layer_index}_zero_point_spline_scaler.txt")

    with open(base_weight_file, 'w') as f:
        base_weight_data = layer.base_weight.detach().cpu().numpy()
        quantized_base_weight, scale, zero_point = quantize_tensor(torch.tensor(base_weight_data), num_bits)
        for value in quantized_base_weight.flatten():
            f.write(f'{int_to_binary(value, num_bits)}\n')
    with open(base_weight_scale_file, 'w') as f:
        f.write(f"{scale}\n")
    with open(base_weight_zero_point_file, 'w') as f:
        f.write(f"{zero_point}\n")

    with open(spline_weight_file, 'w') as f:
        spline_weight_data = layer.spline_weight.detach().cpu().numpy()
        quantized_spline_weight, scale, zero_point = quantize_tensor(torch.tensor(spline_weight_data), num_bits)
        for value in quantized_spline_weight.flatten():
            f.write(f'{int_to_binary(value, num_bits)}\n')
    with open(spline_weight_scale_file, 'w') as f:
        f.write(f"{scale}\n")
    with open(spline_weight_zero_point_file, 'w') as f:
        f.write(f"{zero_point}\n")

    with open(spline_scaler_file, 'w') as f:
        spline_scaler_data = layer.spline_scaler.detach().cpu().numpy()
        quantized_spline_scaler, scale, zero_point = quantize_tensor(torch.tensor(spline_scaler_data), num_bits)
        for value in quantized_spline_scaler.flatten():
            f.write(f'{int_to_binary(value, num_bits)}\n')
    with open(spline_scaler_scale_file, 'w') as f:
        f.write(f"{scale}\n")
    with open(spline_scaler_zero_point_file, 'w') as f:
        f.write(f"{zero_point}\n")

# 從 TXT 文件加載量化後的權重
def binary_to_float(binary_str):
    import struct
    bf = int(binary_str, 2)
    return struct.unpack(">d", struct.pack(">Q", bf))[0]

def binary_to_int(binary_str):
    return int(binary_str, 2)

def read_quantized_file(file_path, num_bits):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    quantized_values = np.array([binary_to_int(v.strip()) for v in lines])
    return quantized_values

def read_scale_zero_point(file_path, is_float):
    with open(file_path, 'r') as f:
        value = f.readline().strip()
    return float(value) if is_float else int(value)

def load_quantized_weights_from_txt(model, layer_index, base_dir, num_bits):
    layer = model.layers[layer_index]
    folder_name = os.path.join(base_dir, f"{num_bits}bits")

    base_weight_file = os.path.join(folder_name, f"layer{layer_index}_base_weight.txt")
    spline_weight_file = os.path.join(folder_name, f"layer{layer_index}_spline_weight.txt")
    spline_scaler_file = os.path.join(folder_name, f"layer{layer_index}_spline_scaler.txt")

    base_weight_scale_file = os.path.join(folder_name, f"layer{layer_index}_scale_base_weight.txt")
    base_weight_zero_point_file = os.path.join(folder_name, f"layer{layer_index}_zero_point_base_weight.txt")

    spline_weight_scale_file = os.path.join(folder_name, f"layer{layer_index}_scale_spline_weight.txt")
    spline_weight_zero_point_file = os.path.join(folder_name, f"layer{layer_index}_zero_point_spline_weight.txt")

    spline_scaler_scale_file = os.path.join(folder_name, f"layer{layer_index}_scale_spline_scaler.txt")
    spline_scaler_zero_point_file = os.path.join(folder_name, f"layer{layer_index}_zero_point_spline_scaler.txt")

    quantized_base_weight = read_quantized_file(base_weight_file, num_bits)
    base_weight_scale = read_scale_zero_point(base_weight_scale_file, is_float=True)
    base_weight_zero_point = read_scale_zero_point(base_weight_zero_point_file, is_float=False)
    layer.base_weight.data = torch.tensor((quantized_base_weight - base_weight_zero_point) * base_weight_scale,
                                          dtype=torch.float32).view_as(layer.base_weight)

    quantized_spline_weight = read_quantized_file(spline_weight_file, num_bits)
    spline_weight_scale = read_scale_zero_point(spline_weight_scale_file, is_float=True)
    spline_weight_zero_point = read_scale_zero_point(spline_weight_zero_point_file, is_float=False)
    layer.spline_weight.data = torch.tensor((quantized_spline_weight - spline_weight_zero_point) * spline_weight_scale,
                                            dtype=torch.float32).view_as(layer.spline_weight)

    if os.path.exists(spline_scaler_file):
        quantized_spline_scaler = read_quantized_file(spline_scaler_file, num_bits)
        spline_scaler_scale = read_scale_zero_point(spline_scaler_scale_file, is_float=True)
        spline_scaler_zero_point = read_scale_zero_point(spline_scaler_zero_point_file, is_float=False)
        layer.spline_scaler.data = torch.tensor(
            (quantized_spline_scaler - spline_scaler_zero_point) * spline_scaler_scale, dtype=torch.float32).view_as(
            layer.spline_scaler)

# 測試量化後的模型性能
def test_quantized_model(model, base_dir, num_bits):
    for i in range(len(model.layers)):
        load_quantized_weights_from_txt(model, i, base_dir, num_bits)

    model.eval()
    with torch.no_grad():
        test_x = torch.rand(1024, 2)
        test_y = model(test_x)
        u = test_x[:, 0]
        v = test_x[:, 1]
        expected_y = u * v
        test_loss = nn.functional.mse_loss(test_y.squeeze(-1), expected_y)
        print(f"Test Loss with {num_bits}-bit quantization: {test_loss.item():.4f}")

# 呼叫函數保存每層的權重，使用16位、8位和4位量化和其他位數
bit_levels = [16, 8, 4, 2]
base_dir = 'weightsMultiplication'
for num_bits in bit_levels:
    for i in range(len(model.layers)):
        save_layer_weights_to_txt(model, i, base_dir, num_bits)

print("Weights and scale/zero points saved to TXT files with quantization in separate folders.")

# 測試不同量化位數的模型
for num_bits in bit_levels:
    print(f"\nTesting {num_bits}-bit quantized model:")
    test_quantized_model(model, base_dir, num_bits)
