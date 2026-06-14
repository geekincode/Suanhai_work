#ifndef MATRIX_CALCULATIONS_HPP
#define MATRIX_CALCULATIONS_HPP

#include <string>
#include <Eigen/Core>

template <typename Derived>
void printEigenObject(const std::string& name,
                      const Eigen::MatrixBase<Derived>& object)
{
    std::cout << "\n" << name << " =\n" << object << std::endl;
}



#endif