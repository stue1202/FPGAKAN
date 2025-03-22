#include <cmath>
#include <iostream>
#include <random>
using namespace std;
typedef float fixed_t;
const int grid_size=5;
const int spline_order=3;
const float scale_base=1.0;
const float scale_spline=1.0;
const int enable_standalone_scale_spline=1;
const float grid_eps=0.02;
const int grid_range[2]={-1, 1};
const int hiden_layers=3;
struct fpgaARR
{
    float *arr;
    int dim1;
    int dim2;
    int dim3;
    void view() {
        this->dim1*=this->dim2;
        this->dim2=this->dim3;
        this->dim3=1;
    }
    int len() {
        return this->dim1*this->dim2*this->dim3;
    }
    float get(int i, int j, int k) const {
        return this->arr[i * this->dim2 * this->dim1  + j * this->dim1 + k];
    }
    void set(int i, int j, int k,float value) const {
        this->arr[i * this->dim2 * this->dim1  + j * this->dim1 + k]=value;
    }
    void shape() {
        cout << "dim3: " << this->dim3 << ", dim2: " << this->dim2 << ", dim1: " << this->dim1 << endl;
    }
    fpgaARR(float *arr, int dim3, int dim2, int dim1) : arr(arr), dim1(dim1), dim2(dim2), dim3(dim3) {}
};
fpgaARR layers_0_base_weight = {
    new float[3 * 2] {-0.6335, -0.2360, -0.6080, -0.3417, 0.6330, 0.5733},
    1,3,2
};
fpgaARR layers_0_spline_weight = {
    new float[3 * 2 * 8] {
        -3.0776e-10, 3.3747e-09, 1.3112e-08, 5.9135e-09, 1.0272e-08, 1.0224e-08, -6.0491e-09, -1.3825e-09,
        1.9554e-09, 2.3645e-08, 3.8890e-08, 1.8446e-08, 2.7702e-09, -6.4991e-09, -9.9835e-09, -1.7277e-09,
        -5.5773e-10, -5.8572e-10, 3.9641e-09, -6.4413e-09, -1.3370e-08, -2.4077e-08, -2.5515e-08, -3.0608e-09,
        5.3521e-11, 1.4518e-10, 2.6038e-10, -1.7279e-11, -3.5076e-10, -7.0528e-10, -5.1637e-10, -9.4657e-11,
        1.1376e-09, 3.2219e-09, -5.5578e-09, -2.2601e-09, -1.2646e-08, -1.3209e-08, 7.8388e-09, 1.7613e-09,
        -1.7365e-10, -3.1993e-09, -8.6954e-09, -5.6934e-09, -2.6869e-09, -1.2755e-11, 3.5855e-09, 8.1429e-10
    },
    3, 2, 8
};
fpgaARR layers_0_spline_scaler = {
    new float[3 * 2] {0.3008, 0.5762, -0.7051, -0.0114, 0.2279, 0.1865},
    1,3,2
};
fpgaARR layers_1_base_weight = {
    new float[3 * 3] {0.3569, -0.1007, 0.2012, 0.2349, 0.0860, -0.2791, 0.4790, -0.3286, -0.4684},
    1,3, 3
};
fpgaARR layers_1_spline_weight = {
    new float[3 * 3 * 8] {
        -1.1268e-09, -3.7672e-09, 2.6320e-08, 5.2638e-08, -1.0122e-08, -2.2428e-08, -1.9127e-08, -2.3668e-09,
        9.5259e-10, -2.9518e-09, -2.5468e-08, -1.9076e-08, 8.4533e-09, 1.5728e-08, 9.0292e-09, 1.4746e-09,
        -2.7368e-10, -2.2573e-09, -5.8904e-09, -3.3389e-10, 7.6338e-09, 4.9206e-09, -2.4099e-10, 6.9739e-12,
        3.5006e-09, 2.3240e-08, -1.5685e-09, -5.0441e-08, -6.6387e-09, 1.9633e-09, 1.2000e-08, 1.7471e-09,
        1.1218e-09, 4.1386e-09, -3.5517e-09, -8.1166e-09, -2.0767e-09, 1.7170e-09, 2.1775e-09, 5.4089e-10,
        1.2399e-09, 6.3154e-09, 1.4040e-08, -2.0015e-08, -3.7039e-08, 9.9632e-10, 1.9312e-08, 2.1235e-09,
        -3.7845e-09, -2.6594e-08, -1.8488e-09, 5.7621e-08, 1.2521e-08, -8.3752e-10, -1.5378e-08, -2.3517e-09,
        6.5864e-09, 2.7618e-08, -1.4092e-08, -5.3193e-08, -2.3915e-08, 1.0108e-08, 1.6772e-08, 3.5362e-09,
        3.0227e-09, 1.6324e-08, 2.8696e-08, -5.9399e-08, -8.0241e-08, 1.5367e-08, 4.3634e-08, 4.1582e-09
    },
    3, 3, 8
};
fpgaARR layers_1_spline_scaler = {
    new float[3 * 3] {-0.5390, 0.3132, -0.0860, -0.3755, -0.0923, -0.2988, 0.3649, -0.5077, -0.5468},
    1, 3, 3
};
fpgaARR layers_2_base_weight = {
    new float[1 * 3] {0.1959, -0.3349, -0.4020},
    1, 1, 3
};
fpgaARR layers_2_spline_weight = {
    new float[1 * 3 * 8] {
        8.1571e-09, -8.6782e-08, -3.5593e-07, -6.4872e-07, 4.2007e-07, 4.6637e-07, 1.3623e-07, 1.0801e-08,
        6.1854e-09, 3.1181e-08, -4.3499e-09, -5.3605e-08, -1.2841e-08, 6.3710e-09, 9.3972e-09, 2.1327e-09,
        -3.7641e-10, -8.5146e-08, -6.2426e-08, 2.2075e-07, 6.2331e-08, -5.3474e-08, -2.4725e-08, -2.9146e-09
    },
    1, 3, 8
};
fpgaARR layers_2_spline_scaler = {
    new float[1 * 3] {0.4120, 0.1307, -0.4507},
    1,1, 3
};
fpgaARR layers_base_weight[hiden_layers]={
    layers_0_base_weight,
    layers_1_base_weight,
    layers_2_base_weight
};
fpgaARR layers_spline_weight[hiden_layers]={
    layers_0_spline_weight,
    layers_1_spline_weight,
    layers_2_spline_weight
};
fpgaARR layers_spline_scaler[hiden_layers]={
    layers_0_spline_scaler,
    layers_1_spline_scaler,
    layers_2_spline_scaler
};
void silu(fpgaARR &x){
    for(int i=0;i<x.dim3;i++){
        for(int j=0;j<x.dim2;j++){
            for(int k=0;k<x.dim1;k++){
                x.set(i,j,k,x.get(i,j,k)/(1.0+exp(-x.get(i,j,k))));
            }
        }
    }
}
void printarr(const fpgaARR &p){
    printf("dim3: %d, dim2: %d, dim1:%d\n", p.dim3, p.dim2, p.dim1);
    for(int i=0;i<p.dim3;i++){
        for (int j = 0; j < p.dim2; ++j) {
            for (int k = 0; k < p.dim1; ++k) {
                cout << p.get(i,j,k) << " ";
            }
        }
        cout << endl;
    }
}
fpgaARR linear(const fpgaARR &x,const fpgaARR &weight,const fpgaARR &x1,const fpgaARR &weight1){//限定二维
    //float tmp[weight.dim2*x.dim2]={0};
    static float tmp[1024*1024]={0};
    fpgaARR output(tmp,1,x.dim2,weight.dim2);
    for (int i = 0; i < x.dim2; ++i) {
        for (int j = 0; j < weight.dim2; ++j) {
            for (int k = 0; k < x.dim1; ++k) {
                output.arr[i*output.dim1+j]+=x.arr[i*x.dim1+k]*weight.arr[j*weight.dim1+k];
            }
        }
    }
    static float tmp1[1024*1024]={0};
    fpgaARR output1(tmp1,1,x1.dim2,weight1.dim2);
    for (int i = 0; i < x1.dim2; ++i) {
        for (int j = 0; j < weight1.dim2; ++j) {
            for (int k = 0; k < x1.dim1; ++k) {
                output1.arr[i*output1.dim1+j]+=x1.arr[i*x.dim1+k]*weight1.arr[j*weight1.dim1+k];
            }
        }
    }
    for(int i=0;i<output.dim3;i++){
        for(int j=0;j<output.dim2;j++){
            for(int k=0;k<output.dim1;k++){
                output.arr[i*output.dim1+j]+=output1.arr[i*output.dim1+j];
            }
        }
    }
    //cout<<"linear"<<endl;
    //output.shape();
    return output;
}
//fpgaARR addarr(fpgaARR x,fpgaARR y){
//    if(x.dim1==y.dim1&&x.dim2==y.dim2){
//        //float tmp[x.len()]={0};
//        static float tmp[1024*2]={0};
//        fpgaARR output(tmp,1,x.dim2,x.dim1);
//        for (int i = 0; i < x.dim3; ++i) {
//            for (int j = 0; j < x.dim2; ++j) {
//                for (int k = 0; k < x.dim1; ++k) {
//                    output.set(i,j,k,x.get(i,j,k)+y.get(i,j,k));
//                }
//            }
//        }
//        return output;
//    }
//}
fpgaARR b_splines(const fpgaARR &x){
    static float t[grid_size + 2 * spline_order + 1]={0};
    fpgaARR grid(t,1,1,grid_size + 2 * spline_order + 1);
    float h = (grid_range[1] - grid_range[0]) / (float)grid_size;
    for (int i = -spline_order; i <= grid_size + spline_order + 1; ++i) {
        grid.arr[i+spline_order] = i * h + grid_range[0];
    }
    static float tmp[1024*3*11]={0};
    fpgaARR bases(tmp,x.dim2,x.dim1,grid.len()-1);
    for (int i = 0; i < x.dim3; ++i) {
        for (int j = 0; j < x.dim2; ++j) {
            for (int k = 0; k < x.dim1; ++k) {
                float target= x.get(i,j,k);
                for(int l=0;l<grid.len()-1;l++){
                    if(target>=grid.arr[l]&&target<grid.arr[l+1]){
                        bases.set(j,k,l,1);
                    }else{
                        bases.set(j,k,l,0);
                    }
                }
            }
        }
    }
    for(int t=1;t<=spline_order;t++){
        for(int i=0;i<bases.dim3;i++){
            for(int j=0;j<bases.dim2;j++){//data
                float target=x.get(0,i,j);
                for(int k=0;k<bases.dim1;k++){//every knots
                    bases.set(i,j,k,(target-grid.arr[k])/(grid.arr[k+t]-grid.arr[k])*bases.get(i,j,k) + (grid.arr[k+t+1]-target)/(grid.arr[k+t+1]-grid.arr[k+1])*bases.get(i,j,k+1));
                }
            }
        }
        bases.dim1--;
    }
    bases.view();
    return bases;
}

void KanLayer(fpgaARR &x,int layer_now){
    cout<<"layer: "<<layer_now<<endl;
    silu(x);
    layers_spline_weight[layer_now].view();
    fpgaARR B=b_splines(x);
    cout<<"ck1: "<<layer_now<<endl;
    x.shape();
    layers_base_weight[layer_now].shape();
    B.shape();
    layers_spline_weight[layer_now].shape();
    x=linear(x, layers_base_weight[layer_now],B,layers_spline_weight[layer_now]);
}
void KAN(fpgaARR &x,int hiden_layers_number){
    for(int i=0;i<hiden_layers_number;i++){
        KanLayer(x,i);
    }
    
}
int main(){
    //random_test_data
    float arr1D[2048] = {0};
    random_device rd;                       
    mt19937 gen(rd());                      
    uniform_real_distribution<> dis(-1, 1); 
    for (int i = 0; i < 2048; ++i) {
        arr1D[i] = dis(gen);
    }
    fpgaARR x(arr1D, 1,1024,2);
    
    //test
    //for(int i=0;i<1024;i++){
    //    printf("%f * %f = %f\n",x.arr[i*x.dim1+0],x.arr[i*x.dim1+1],x.arr[i*x.dim1+0]*x.arr[i*x.dim1+1]);        
    //}
    KAN(x,hiden_layers);
    printarr(x);
    return 0;
}