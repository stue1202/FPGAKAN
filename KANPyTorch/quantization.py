import numpy as np
import torch

# Load the trained model
model = torch.load('model/kan_multiple_model.pth')
model.eval()


# Quantize the weights to fixed-point format with dynamic scaling
def quantize_weight(weight, scale):
    return (weight * scale).round().int().numpy()


# Convert to 16-bit signed hex
def to_hex_str(arr):
    return [format(x & 0xFFFF, '04X') for x in arr]


# Function to find dynamic scale
def find_dynamic_scale(weight, target_range=32767):
    max_val = torch.max(torch.abs(weight)).item()
    if max_val == 0:
        return 1
    return target_range / max_val


# Extract and quantize weights
layer_base_weights = []
layer_spline_weights = []
for i, layer in enumerate(model.layers):
    # Print statistics of weights before quantization
    print(f"Layer {i} base_weight min: {layer.base_weight.min()}, max: {layer.base_weight.max()}")
    print(f"Layer {i} spline_weight min: {layer.spline_weight.min()}, max: {layer.spline_weight.max()}")

    # Determine dynamic scales
    base_scale = find_dynamic_scale(layer.base_weight)
    spline_scale = find_dynamic_scale(layer.spline_weight)

    print(f"Layer {i} base_scale: {base_scale}")
    print(f"Layer {i} spline_scale: {spline_scale}")

    # Quantize weights with dynamic scales
    base_weight = quantize_weight(layer.base_weight, scale=base_scale).flatten()
    spline_weight = quantize_weight(layer.spline_weight, scale=spline_scale).flatten()

    # Check if spline weights are being quantized to zero
    if np.all(spline_weight == 0):
        print(f"Warning: All spline weights in layer {i} are quantized to zero.")

    # Save weights to files
    with open(f'weight2/base_weight_layer_{i}.txt', 'w') as f:
        f.write('\n'.join(to_hex_str(base_weight)))
    with open(f'weight2/spline_weight_layer_{i}.txt', 'w') as f:
        f.write('\n'.join(to_hex_str(spline_weight)))
