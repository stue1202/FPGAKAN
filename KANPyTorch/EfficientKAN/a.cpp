#include <iostream>
#include <cmath>
#include <cstring> // 用于 memset

// 定义 fpgaarr 结构体
struct fpgaarr {
    float *arr;
    int dim;  // 總维度
    int dim1; // 1维长度
    int dim2; // 2维宽度
    int dim3; // 3维层数
};

// SiLU 激活函数
template <typename T>
T silu(T x) {
    return x / (1.0 + exp(-x));
}

// 计算 B-Spline 基函数 (修改后的版本)
template <typename T, int IN_FEATURES, int GRID_SIZE, int SPLINE_ORDER>
void calculate_b_spline_bases_batch(
    const fpgaarr& x_fpga,        // 输入 x,  二维 (BATCH_SIZE x IN_FEATURES)
    const fpgaarr& grid_fpga,     // 输入 grid, 二维 (IN_FEATURES x (GRID_SIZE + 2 * SPLINE_ORDER + 1))
    fpgaarr& bases_fpga          // 输出 bases, 三维 (BATCH_SIZE x IN_FEATURES x (GRID_SIZE + SPLINE_ORDER))
) {
    // 1. 检查维度是否正确 (非常重要)
     if (x_fpga.dim != 2 || grid_fpga.dim != 2 || bases_fpga.dim != 3) {
        std::cerr << "Error: Incorrect dimensions for fpgaarr structures." << std::endl;
        return; // 或者抛出异常
    }
    //x的第2维长度要等于IN_FEATURES
    if (x_fpga.dim2 != IN_FEATURES) {
        std::cerr << "Error: x_fpga.dim2 != IN_FEATURES" << std::endl;
        return;
    }
    //grid的第1维长度要等于IN_FEATURES
    if (grid_fpga.dim1 != IN_FEATURES) {
        std::cerr << "Error: grid_fpga.dim1 != IN_FEATURES" << std::endl;
        return;
    }
    //grid的第2维长度要等于GRID_SIZE + 2 * SPLINE_ORDER + 1
    if (grid_fpga.dim2 != GRID_SIZE + 2 * SPLINE_ORDER + 1) {
        std::cerr << "Error: grid_fpga.dim2 != GRID_SIZE + 2 * SPLINE_ORDER + 1" << std::endl;
        return;
    }
    //bases的第1维长度要等于BATCH_SIZE
    if (bases_fpga.dim1 != x_fpga.dim1) {
        std::cerr << "Error: bases_fpga.dim1 != x_fpga.dim1 (BATCH_SIZE)" << std::endl;
        return;
    }
    //bases的第2维长度要等于IN_FEATURES
    if (bases_fpga.dim2 != IN_FEATURES) {
        std::cerr << "Error: bases_fpga.dim2 != IN_FEATURES" << std::endl;
        return;
    }
    //bases的第3维长度要等于GRID_SIZE + SPLINE_ORDER
    if (bases_fpga.dim3 != GRID_SIZE + SPLINE_ORDER) {
        std::cerr << "Error: bases_fpga.dim3 != GRID_SIZE + SPLINE_ORDER" << std::endl;
        return;
    }

    int BATCH_SIZE = x_fpga.dim1; // 从 x_fpga 获取 BATCH_SIZE


    // (1) 0 阶 B-spline 初始化
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < IN_FEATURES; i++) {
            for (int j = 0; j < (GRID_SIZE + SPLINE_ORDER); j++) {
                // 使用指针和偏移量访问数组元素
                T x_val = x_fpga.arr[b * IN_FEATURES + i];
                T grid_val_left = grid_fpga.arr[i * (GRID_SIZE + 2 * SPLINE_ORDER + 1) + j];
                T grid_val_right = grid_fpga.arr[i * (GRID_SIZE + 2 * SPLINE_ORDER + 1) + j + 1];

                if (x_val >= grid_val_left && x_val < grid_val_right) {
                    bases_fpga.arr[b * IN_FEATURES * (GRID_SIZE + SPLINE_ORDER) + i * (GRID_SIZE + SPLINE_ORDER) + j] = T(1.0);
                } else {
                    bases_fpga.arr[b * IN_FEATURES * (GRID_SIZE + SPLINE_ORDER) + i * (GRID_SIZE + SPLINE_ORDER) + j] = T(0.0);
                }
            }
        }
    }

    // (2) Cox–de Boor 递推
    for (int k = 1; k <= SPLINE_ORDER; k++) {
        for (int b = 0; b < BATCH_SIZE; b++) {
            for (int i = 0; i < IN_FEATURES; i++) {
                for (int j = 0; j < (GRID_SIZE + SPLINE_ORDER) - k; j++) {
                    // 计算 grid 数组的偏移量
                    int grid_offset = i * (GRID_SIZE + 2 * SPLINE_ORDER + 1);

                    T denom_left = (grid_fpga.arr[grid_offset + j + k] - grid_fpga.arr[grid_offset + j]);
                    T denom_right = (grid_fpga.arr[grid_offset + j + k + 1] - grid_fpga.arr[grid_offset + j + 1]);

                    T x_val = x_fpga.arr[b * IN_FEATURES + i]; // 获取 x[b][i]

                    T left_coeff;
                    if (denom_left != T(0.0)) {
                        left_coeff = (x_val - grid_fpga.arr[grid_offset + j]) / denom_left;
                    } else {
                        left_coeff = T(0.0);
                    }

                    T right_coeff;
                    if (denom_right != T(0.0)) {
                        right_coeff = (grid_fpga.arr[grid_offset + j + k + 1] - x_val) / denom_right;
                    } else {
                        right_coeff = T(0.0);
                    }

                    // 计算 bases 数组的偏移量
                    int bases_offset = b * IN_FEATURES * (GRID_SIZE + SPLINE_ORDER) + i * (GRID_SIZE + SPLINE_ORDER);

                    // 递推计算
                    bases_fpga.arr[bases_offset + j] = left_coeff * bases_fpga.arr[bases_offset + j]
                                                     + right_coeff * bases_fpga.arr[bases_offset + j + 1];
                }
            }
        }
    }
}

// 通用的 linear_layer
template <typename T, int IN_COLS, int OUT_COLS>
void linear_layer(T input[IN_COLS], T weight[OUT_COLS][IN_COLS], T output[OUT_COLS]) {
    #pragma HLS INLINE
    for (int j = 0; j < OUT_COLS; j++) {
        T sum = 0;
        for (int i = 0; i < IN_COLS; i++) {
            sum += input[i] * weight[j][i];
        }
        output[j] = sum;
    }
}

// 计算 base_output
template <typename T, int IN_FEATURES, int OUT_FEATURES>
void compute_base_output(T x[IN_FEATURES], T weight[OUT_FEATURES][IN_FEATURES], T output[OUT_FEATURES]) {
    T activated_x[IN_FEATURES];

    // SiLU 激活函数
    for (int i = 0; i < IN_FEATURES; i++) {
        activated_x[i] = silu(x[i]);
    }

    // 调用通用 linear_layer
    linear_layer<T, IN_FEATURES, OUT_FEATURES>(activated_x, weight, output);
}

// 计算 B-Spline spline_output
template <typename T, int IN_FEATURES, int GRID_SIZE, int SPLINE_ORDER, int OUT_FEATURES>
void compute_spline_output(
    T x[IN_FEATURES],
    T b_spline_basis[IN_FEATURES][GRID_SIZE + SPLINE_ORDER],
    T spline_weight[OUT_FEATURES][IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)],
    T output[OUT_FEATURES]
) {
    T bases[IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)];

    // 展平成 1D 矩阵
    int idx = 0;
    for (int i = 0; i < IN_FEATURES; i++) {
        for (int j = 0; j < (GRID_SIZE + SPLINE_ORDER); j++) {
            bases[idx++] = b_spline_basis[i][j];
        }
    }

    // 调用通用 linear_layer
    linear_layer<T, IN_FEATURES * (GRID_SIZE + SPLINE_ORDER), OUT_FEATURES>(bases, spline_weight, output);
}

// KANLayer forward() 版本 (修改参数类型)
template <typename T, int IN_FEATURES, int GRID_SIZE, int SPLINE_ORDER, int OUT_FEATURES>
void KANLayer(
    T* x,  // 改为指针
    T* base_weight, // 改为指针
    T* spline_weight, // 改为指针
    T* b_spline_basis, // 改为指针
    T* output // 改为指针
) {
    T base_output[OUT_FEATURES], spline_output[OUT_FEATURES];

    // 使用指针和偏移量访问数组元素
    compute_base_output<T, IN_FEATURES, OUT_FEATURES>(x, (T(*)[IN_FEATURES])base_weight, base_output);
    compute_spline_output<T, IN_FEATURES, GRID_SIZE, SPLINE_ORDER, OUT_FEATURES>(
        x,
        (T(*)[GRID_SIZE + SPLINE_ORDER])b_spline_basis,
        (T(*)[IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)])spline_weight,
        spline_output
    );

    // 加总两个输出
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = base_output[i] + spline_output[i];
    }
}

// 主函数 (修改后以适应 fpgaarr)
int main() {
    constexpr int IN_FEATURES = 5;
    constexpr int OUT_FEATURES = 3;
    constexpr int GRID_SIZE = 10;
    constexpr int SPLINE_ORDER = 3;
    constexpr int BATCH_SIZE = 4;  // 假设批次大小

    // --- 初始化 fpgaarr 结构体 ---
    fpgaarr x_fpga, grid_fpga, bases_fpga;

    // x_fpga (二维: BATCH_SIZE x IN_FEATURES)
    x_fpga.dim = 2;
    x_fpga.dim1 = BATCH_SIZE;
    x_fpga.dim2 = IN_FEATURES;
    x_fpga.dim3 = 0; // 二维数组, dim3 无意义
    x_fpga.arr = new float[BATCH_SIZE * IN_FEATURES];
    // 初始化 x_fpga.arr (示例数据)
    float x_data[BATCH_SIZE][IN_FEATURES] = {
        {0.5, -0.2, 0.3, -0.1, 0.8},
        {0.6, -0.3, 0.4, -0.2, 0.9},
        {0.4, -0.1, 0.2,  0.0, 0.7},
        {0.7, -0.4, 0.5, -0.3, 1.0}
    };
    std::memcpy(x_fpga.arr, x_data, sizeof(x_data));

    // grid_fpga (二维: IN_FEATURES x (GRID_SIZE + 2 * SPLINE_ORDER + 1))
    grid_fpga.dim = 2;
    grid_fpga.dim1 = IN_FEATURES;
    grid_fpga.dim2 = GRID_SIZE + 2 * SPLINE_ORDER + 1;
    grid_fpga.dim3 = 0; // 二维数组, dim3 无意义
    grid_fpga.arr = new float[IN_FEATURES * (GRID_SIZE + 2 * SPLINE_ORDER + 1)];
    // 初始化 grid_fpga.arr (示例: 均匀网格)
    for (int i = 0; i < IN_FEATURES; i++) {
        for (int j = 0; j < GRID_SIZE + 2 * SPLINE_ORDER + 1; j++) {
            grid_fpga.arr[i * (GRID_SIZE + 2 * SPLINE_ORDER + 1) + j] = (float)j / (GRID_SIZE + 2 * SPLINE_ORDER);
        }
    }

    // bases_fpga (三维: BATCH_SIZE x IN_FEATURES x (GRID_SIZE + SPLINE_ORDER))
    bases_fpga.dim = 3;
    bases_fpga.dim1 = BATCH_SIZE;
    bases_fpga.dim2 = IN_FEATURES;
    bases_fpga.dim3 = GRID_SIZE + SPLINE_ORDER;
    bases_fpga.arr = new float[BATCH_SIZE * IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)];
    std::memset(bases_fpga.arr, 0, sizeof(float) * BATCH_SIZE * IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)); // 初始化为 0

    // --- 调用 calculate_b_spline_bases_batch ---
    calculate_b_spline_bases_batch<float, IN_FEATURES, GRID_SIZE, SPLINE_ORDER>(x_fpga, grid_fpga, bases_fpga);


    // --- 准备 KANLayer 的参数 ---
    // (注意: 这里我们只创建了原始数组, 没有使用 fpgaarr 包裹它们)
    float base_weight[OUT_FEATURES][IN_FEATURES] = {
        {0.1, 0.2, 0.3, 0.4, 0.5},
        {0.2, 0.3, 0.4, 0.5, 0.6},
        {0.3, 0.4, 0.5, 0.6, 0.7}
    };
    float spline_weight[OUT_FEATURES][IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)] = {0}; // 初始化为 0
    float output[OUT_FEATURES] = {0}; // 初始化 output

    // --- 调用 KANLayer (重要: 传递指针) ---
    // 因为 KANLayer, compute_base_output, compute_spline_output, linear_layer 没有修改,
    // 所以我们需要传递原始数组的指针给它们。
      for (int i = 0; i < BATCH_SIZE; ++i) {
        KANLayer<float, IN_FEATURES, GRID_SIZE, SPLINE_ORDER, OUT_FEATURES>(
            &x_fpga.arr[i*IN_FEATURES],           // 傳入x
            (float*)base_weight,     // 傳入指針
            (float*)spline_weight,   // 傳入指針
            &bases_fpga.arr[i*IN_FEATURES*(GRID_SIZE + SPLINE_ORDER)], // 傳入 bases
            output                  // 傳入指針
        );

        std::cout << "Final KAN Output (Batch " << i << "): ";
        for (int j = 0; j < OUT_FEATURES; j++) {
            std::cout << output[j] << " ";
        }
        std::cout << std::endl;
     }
    // --- 释放内存 
    delete[] x_fpga.arr;
    delete[] grid_fpga.arr;
    delete[] bases_fpga.arr;

    return 0;
}