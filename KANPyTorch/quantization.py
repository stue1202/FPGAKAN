import torch

# Load the trained model
model = torch.load('model/kan_multiple_model.pth')
model.eval()


# Quantize the weights to fixed-point format
def quantize_weight(weight, scale=256):
    return (weight * scale).round().int().numpy()


# Convert to 16-bit signed hex
def to_hex_str(arr):
    return [format(x & 0xFFFF, '04X') for x in arr]


# Extract and quantize weights
layer_base_weights = []
layer_spline_weights = []
for i, layer in enumerate(model.layers):
    base_weight = quantize_weight(layer.base_weight).flatten()
    spline_weight = quantize_weight(layer.spline_weight).flatten()

    # Save weights to files
    with open(f'weight2/base_weight_layer_{i}.txt', 'w') as f:
        f.write('\n'.join(to_hex_str(base_weight)))
    with open(f'weight2/spline_weight_layer_{i}.txt', 'w') as f:
        f.write('\n'.join(to_hex_str(spline_weight)))
