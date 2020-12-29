// lvalue and rvalue
//  - lvalue: An object that occypies some identifiable location in memory.
//  - rvalue: Any object that is not a lvalue.
// 
// Misconceptions:
//     function or operator always yields rvalues.     **(wrong)**
//     lvalues are modifiable.                         **(wrong)**
//     rvalues are not modifiable.                     **(wrong)**
// c++11  Rvalue Reference
//     1. Move sementics
//         - most useful place for rvalue reference is overloading copy constructor and copy assigment operator.
//         - x &x::operator= (x const &rhs);
//           x &x::operator=(x &&rhs);
//         - boVector(const boVector &rhs){       //copy construct
//                  size = rhs.size;
//                  arr_ = new double[size];
//                  for(int i =0; i<size; i++){arr_[i] = rhs.arr_[i];}  //deep copy
//              }
//           boVector(boVector &&rhs){           //Move constructor
//     	            size = rhs.size;
//     	            arr_ = rhs.arr_;
//     	            rhs.arr_ = nullptr;          // avoid change value out of class
//              }
//     2. Perfect forwarding

#include <iostream>
using namespace std;

void printInput(int &i){
    cout << "lvalue reference: " << i << endl;
}

void printInput(int &&i){
    cout << "rvalue reference: " << i << endl;
}

int main(){
    // int a = 5;
    // printInput(a);
    // printInput(5);

    const int i = 6;
    const_cast<int&>(i) = 5;
    int j = 7;
    // static_cast<const int&>(j) = 8;
}