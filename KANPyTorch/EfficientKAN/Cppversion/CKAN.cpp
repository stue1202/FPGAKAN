#include <iostream>
#include <cmath>

// SiLU 激活函數
template <typename T>
T silu(T x) {
    return x / (1.0 + exp(-x));
}
template <typename T, int IN_FEATURES, int GRID_SIZE, int SPLINE_ORDER>
void calculate_b_spline_bases(
    T x[IN_FEATURES], 
    T grid[IN_FEATURES][GRID_SIZE + 2 * SPLINE_ORDER + 1], 
    T bases[IN_FEATURES][GRID_SIZE + SPLINE_ORDER]
) {
    #pragma HLS ARRAY_PARTITION variable=grid complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bases complete dim=2

    // 初始化 0 階 B-Spline
    for (int i = 0; i < IN_FEATURES; i++) {
        for (int j = 0; j < GRID_SIZE + SPLINE_ORDER; j++) {
            bases[i][j] = (x[i] >= grid[i][j] && x[i] < grid[i][j + 1]) ? 1.0 : 0.0;
        }
    }

    // 遞推計算 B-Spline
    for (int k = 1; k <= SPLINE_ORDER; k++) {
        for (int i = 0; i < IN_FEATURES; i++) {
            for (int j = 0; j < GRID_SIZE + SPLINE_ORDER - k; j++) {
                T left = (x[i] - grid[i][j]) / (grid[i][j + k] - grid[i][j]);
                T right = (grid[i][j + k + 1] - x[i]) / (grid[i][j + k + 1] - grid[i][j + 1]);

                bases[i][j] = left * bases[i][j] + right * bases[i][j + 1];
            }
        }
    }
}
// 通用的 `linear_layer`，適用於任意 `IN_COLS` 和 `OUT_COLS`
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

// 計算 `base_output`
template <typename T, int IN_FEATURES, int OUT_FEATURES>
void compute_base_output(T x[IN_FEATURES], T weight[OUT_FEATURES][IN_FEATURES], T output[OUT_FEATURES]) {
    T activated_x[IN_FEATURES];
    
    // SiLU 激活函數
    for (int i = 0; i < IN_FEATURES; i++) {
        activated_x[i] = silu(x[i]);
    }

    // 呼叫通用 `linear_layer`
    linear_layer<T, IN_FEATURES, OUT_FEATURES>(activated_x, weight, output);
}

// 計算 B-Spline `spline_output`
template <typename T, int IN_FEATURES, int GRID_SIZE, int SPLINE_ORDER, int OUT_FEATURES>
void compute_spline_output(
    T x[IN_FEATURES], 
    T b_spline_basis[IN_FEATURES][GRID_SIZE + SPLINE_ORDER], 
    T spline_weight[OUT_FEATURES][IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)], 
    T output[OUT_FEATURES]
) {
    T bases[IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)];

    // 展平成 1D 矩陣
    int idx = 0;
    for (int i = 0; i < IN_FEATURES; i++) {
        for (int j = 0; j < (GRID_SIZE + SPLINE_ORDER); j++) {
            bases[idx++] = b_spline_basis[i][j];
        }
    }

    // 呼叫通用 `linear_layer`
    linear_layer<T, IN_FEATURES * (GRID_SIZE + SPLINE_ORDER), OUT_FEATURES>(bases, spline_weight, output);
}

// KANLayer `forward()` 版本
template <typename T, int IN_FEATURES, int GRID_SIZE, int SPLINE_ORDER, int OUT_FEATURES>
void KANLayer(
    T x[IN_FEATURES], 
    T base_weight[OUT_FEATURES][IN_FEATURES], 
    T spline_weight[OUT_FEATURES][IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)], 
    T b_spline_basis[IN_FEATURES][GRID_SIZE + SPLINE_ORDER], 
    T output[OUT_FEATURES]
) {
    T base_output[OUT_FEATURES], spline_output[OUT_FEATURES];

    compute_base_output<T, IN_FEATURES, OUT_FEATURES>(x, base_weight, base_output);
    compute_spline_output<T, IN_FEATURES, GRID_SIZE, SPLINE_ORDER, OUT_FEATURES>(x, b_spline_basis, spline_weight, spline_output);

    // 加總兩個輸出
    for (int i = 0; i < OUT_FEATURES; i++) {
        output[i] = base_output[i] + spline_output[i];
    }
}

// 測試主函數
int main() {
    constexpr int IN_FEATURES = 5;
    constexpr int OUT_FEATURES = 3;
    constexpr int GRID_SIZE = 10;
    constexpr int SPLINE_ORDER = 3;

    // 測試輸入
    float x[IN_FEATURES] = {0.5, -0.2, 0.3, -0.1, 0.8};

    // 初始化 base weight
    float base_weight[OUT_FEATURES][IN_FEATURES] = {
        {0.1, 0.2, 0.3, 0.4, 0.5}, 
        {0.2, 0.3, 0.4, 0.5, 0.6}, 
        {0.3, 0.4, 0.5, 0.6, 0.7}
    };

    // 初始化 spline weight
    float spline_weight[OUT_FEATURES][IN_FEATURES * (GRID_SIZE + SPLINE_ORDER)] = {0};

    // 初始化 B-Spline 基函數
    float b_spline_basis[IN_FEATURES][GRID_SIZE + SPLINE_ORDER] = {0};

    float output[OUT_FEATURES];

    // 呼叫 KANLayer
    KANLayer<float, IN_FEATURES, GRID_SIZE, SPLINE_ORDER, OUT_FEATURES>(
        x, base_weight, spline_weight, b_spline_basis, output
    );

    // 顯示輸出
    std::cout << "Final KAN Output: ";
    for (int i = 0; i < OUT_FEATURES; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
