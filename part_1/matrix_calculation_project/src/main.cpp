#include <iostream>
#include <Eigen/Core>
#include <Eigen/LU>
#include "MatrixCalculations.hpp"

int main()
{
    Eigen::Matrix<double, 10, 10> A;
    Eigen::Matrix<double, 10, 10> B;

    A << Eigen::Matrix<double, 10, 10>::Random();

    B.setIdentity();

    printEigenObject("A.transpose() * B", A.transpose() * B);

    
    Eigen::Matrix<double, 5, 5> M;
    Eigen::Matrix<double, 5, 5> b;

    M << Eigen::Matrix<double, 5, 5>::Random();
    b.setIdentity();


    printEigenObject("M * b", M * b);

    return 0;
}

