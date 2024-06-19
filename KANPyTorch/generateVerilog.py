import os


def read_quantized_weights(folder_path):
    weights = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            layer_name = file_name.replace(".csv", "").replace(".", "_")
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                binary_data = [line.strip() for line in f]
                weights[layer_name] = binary_data
    return weights


def generate_verilog(weights, num_bits=8):
    verilog_code = """module KAN(
    input wire clk,
    input wire rst,
    input wire [31:0] x1,
    input wire [31:0] x2,
    output wire [31:0] y
);

// Intermediate signals
wire [31:0] layer0_out_0;
wire [31:0] layer0_out_1;
wire [31:0] layer0_out_2;
wire [31:0] layer1_out_0;
wire [31:0] layer1_out_1;
wire [31:0] layer1_out_2;
wire [31:0] layer2_out;

"""

    # 宣告所有權重變量
    for layer_name, binary_data in weights.items():
        for i, binary_line in enumerate(binary_data):
            verilog_code += f"reg [{num_bits - 1}:0] {layer_name}_{i};\n"

    verilog_code += "\ninitial begin\n"
    # 初始化所有權重變量
    for layer_name, binary_data in weights.items():
        for i, binary_line in enumerate(binary_data):
            verilog_code += f"    {layer_name}_{i} = {num_bits}'b{binary_line};\n"
    verilog_code += "end\n\n"

    verilog_code += """
// Layer 0: Input to hidden
assign layer0_out_0 = x1 * layers_0_base_weight_0 + x2 * layers_0_base_weight_1;
assign layer0_out_1 = x1 * layers_0_base_weight_2 + x2 * layers_0_base_weight_3;
assign layer0_out_2 = x1 * layers_0_base_weight_4 + x2 * layers_0_base_weight_5;

// Layer 1: Hidden to hidden
assign layer1_out_0 = layer0_out_0 * layers_1_base_weight_0 + layer0_out_1 * layers_1_base_weight_1 + layer0_out_2 * layers_1_base_weight_2;
assign layer1_out_1 = layer0_out_0 * layers_1_base_weight_3 + layer0_out_1 * layers_1_base_weight_4 + layer0_out_2 * layers_1_base_weight_5;
assign layer1_out_2 = layer0_out_0 * layers_1_base_weight_6 + layer0_out_1 * layers_1_base_weight_7 + layer0_out_2 * layers_1_base_weight_8;

// Layer 2: Hidden to output
assign layer2_out = layer1_out_0 * layers_2_base_weight_0 + layer1_out_1 * layers_2_base_weight_1 + layer1_out_2 * layers_2_base_weight_2;

assign y = layer2_out;

endmodule
"""
    return verilog_code


if __name__ == "__main__":
    folder_path = "model/quantized_weights"
    num_bits = 8  # 可以根據實際需要調整量化位數
    weights = read_quantized_weights(folder_path)
    verilog_code = generate_verilog(weights, num_bits)

    with open("logic/KANDigitalDesign.v", "w") as f:
        f.write(verilog_code)
    print("Verilog code generated and saved.")
