
#include <iostream>
using namespace std;
/*
* @Function:
template<typename T,size_t M,size_t N>
void fnPrint2DArray(T (&arr)[M][N])
* @Brief: Print the  content of 2D Array.
* @Input: 2D array.
* @Output: None.
*/
template<typename T,size_t M,size_t N>
void fnPrint2DArray(T (&arr)[M][N], T (&b)[M][N])
{
    for(size_t i=0;i<M;++i)
    {
        for(size_t j=0; j<N; ++j)
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
    // Print 2D array.
    fnPrint2DArray(arr2D,B);
    return 0;
}