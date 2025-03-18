
#include <iostream>
using namespace std;
typedef float fixed_t;

struct fpgaARR
{
    float *arr;
    int dim;
    int dim1;
    int dim2;
    int dim3;
};
void build_grid(const int grid_range[2], const int grid[IN_FEATURES * (GRID_SIZE + 2 * SPLINE_ORDER + 1)]) {
    T h = (grid_range[1] - grid_range[0]) / GRID_SIZE;

    for (int i = 0; i < IN_FEATURES; ++i) {
        for (int j = -SPLINE_ORDER; j <= GRID_SIZE + SPLINE_ORDER; ++j) {
            grid[i * (GRID_SIZE + 2 * SPLINE_ORDER + 1) + (j + SPLINE_ORDER)] = j * h + grid_range[0];
        }
    }
}
template<typename T,size_t row,size_t col>
void fnPrint2DArray(T (&arr)[row][col])
{
    for(size_t i=0;i<row;++i)
    {
        for(size_t j=0; j<col; ++j)
        {
            cout << arr[i][j] << " ";
        }
        cout << endl;
    }
}


int main(int argc, char * argv[])
{
    int arr2D[2][2] = { {1,2},{3,4}}; // Create 2D array.
    int B[2][2] = { {1,2},{3,4}}; // Create 2D array.
    fnPrint2DArray(arr2D);
    cout<<B[0][0]<<endl;
    return 0;
}