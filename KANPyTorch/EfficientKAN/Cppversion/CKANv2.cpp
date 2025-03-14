#include <iostream>
//#include <ap_fixed.h>

// 定義資料類型
//typedef ap_fixed<16, 8> fixed_t; // 16 位固定點，8 位整數部分
typedef float fixed_t;  // testing

#define MAX_FEATURES 10
#define GRID_SIZE 20
#define SPLINE_ORDER 3
#define IN_ROWS 2
#define IN_COLS 2
#define OUT_COLS 3
#define BATCH_SIZE 2
#define IN_FEATURES 2
#define OUT_FEATURES 3

// 線性層函數
void linear(fixed_t input[IN_ROWS][IN_COLS],
            fixed_t weight[OUT_COLS][IN_COLS],
            fixed_t output[IN_ROWS][OUT_COLS]) {
    // 矩陣乘法
    for (int i = 0; i < IN_ROWS; ++i) {
        for (int j = 0; j < OUT_COLS; ++j) {
            fixed_t sum = 0;
            for (int k = 0; k < IN_COLS; ++k) {
                sum += input[i][k] * weight[j][k];
            }
            output[i][j] = sum;
        }
    }
}

void view(float input[BATCH_SIZE][IN_FEATURES][OUT_FEATURES],
          float output[BATCH_SIZE][IN_FEATURES * OUT_FEATURES]) {
    #pragma HLS ARRAY_PARTITION variable=input complete dim=2
    #pragma HLS ARRAY_PARTITION variable=output complete dim=2

    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < IN_FEATURES; j++) {
            for (int k = 0; k < OUT_FEATURES; k++) {
                output[i][j * OUT_FEATURES + k] = input[i][j][k];
            }
        }
    }
}

void silu(fixed_t input[IN_ROWS], fixed_t output[IN_ROWS]) {
    #pragma HLS PIPELINE
    for (int i = 0; i < IN_ROWS; ++i) {
        output[i] = input[i] * input[i] / (1 + abs(input[i]));
    }
}

void KANLayer(fixed_t input[IN_ROWS], fixed_t output[OUT_COLS]) {
    fixed_t base_activation[IN_ROWS][IN_COLS] = {0}; // Placeholder for activation
    fixed_t base_weight[OUT_COLS][IN_COLS] = {0}; // Placeholder for weights
    fixed_t base_output[IN_ROWS][OUT_COLS] = {0};
    linear(base_activation, base_weight, base_output);

    fixed_t scaled_spline_weight[OUT_COLS][IN_FEATURES * OUT_FEATURES] = {0};
    fixed_t b_spline_out[IN_ROWS][IN_FEATURES * OUT_FEATURES] = {0};
    view(base_activation, b_spline_out);
    
    fixed_t spline_output[IN_ROWS][OUT_COLS] = {0};
    linear(b_spline_out, scaled_spline_weight, spline_output);

    for (int i = 0; i < OUT_COLS; i++) {
        output[i] = base_output[0][i] + spline_output[0][i];
    }
}

void calculate_b_spline_bases(
    fixed_t x[MAX_FEATURES], 
    fixed_t grid[MAX_FEATURES][GRID_SIZE + 2 * SPLINE_ORDER + 1], 
    fixed_t bases[MAX_FEATURES][GRID_SIZE + SPLINE_ORDER]) {

    #pragma HLS ARRAY_PARTITION variable=grid complete dim=2
    #pragma HLS ARRAY_PARTITION variable=bases complete dim=2
    
    // 初始化 0 階 B-Spline
    for (int i = 0; i < MAX_FEATURES; i++) {
        for (int j = 0; j < GRID_SIZE + SPLINE_ORDER; j++) {
            if (x[i] >= grid[i][j] && x[i] < grid[i][j+1]) {
                bases[i][j] = 1.0;
            } else {
                bases[i][j] = 0.0;
            }
        }
    }

    // 遞推計算 B-Spline 基函數
    for (int k = 1; k <= SPLINE_ORDER; k++) {
        for (int i = 0; i < MAX_FEATURES; i++) {
            for (int j = 0; j < GRID_SIZE + SPLINE_ORDER - k; j++) {
                fixed_t left  = (x[i] - grid[i][j]) / (grid[i][j + k] - grid[i][j]);
                fixed_t right = (grid[i][j + k + 1] - x[i]) / (grid[i][j + k + 1] - grid[i][j + 1]);

                bases[i][j] = left * bases[i][j] + right * bases[i][j + 1];
            }
        }
    }
}

// 主函數
int main() {
    fixed_t input[IN_ROWS] = {0.3, -0.5};  // 測試輸入
    fixed_t hidden1[OUT_COLS], hidden2[OUT_COLS], output[OUT_COLS];

    KANLayer(input, hidden1);  // 第一層計算
    KANLayer(hidden1, hidden2);  // 第二層計算
    KANLayer(hidden2, output);  // 第三層計算

    std::cout << "Final KAN Output: ";
    for (int i = 0; i < OUT_COLS; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
