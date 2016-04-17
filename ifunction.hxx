#ifndef __IFUNCTION_HXX__
#define __IFUNCTION_HXX__ 

#include <Eigen/Dense>

template <int n>
struct IFunction { 
    typedef Matrix<double, n, 1> Domain;
    virtual _full_gradient(Domain x) = 0;
    virtual _partial_gradient(Domain x) = 0;
    virtual double operator()(const Domain & x) = 0;
};

#endif
