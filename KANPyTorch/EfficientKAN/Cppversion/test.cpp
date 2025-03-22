#include <iostream>
using namespace std;

void test(int &a) {
    int b = 7;
    int* ptr = &b;
    a = *ptr; 
}
int main() {
int a=5;
   test(a);
   cout<<a;
    return 0;
}
