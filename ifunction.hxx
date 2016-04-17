#ifndef __IFUNCTION_HXX__
#define __IFUNCTION_HXX__ 

#include <Eigen/Dense>

/*
 * IFunction is an interface for what a function needs to contain
 * in order to implement SGD, SVRGD, SAGA, and Katyusha
 *
 */

using namespace Eigen;

namespace Optimastic { 

template <int n>
struct IFunction { 
    static const int Dimension = n; 
    typedef Matrix<double, n, 1> Domain;

    // _partial_gradient takes in a gradient vector and accumulates 
    // the ith partial derviative into this vector; this is to avoid 
    // making more temporaries and copies of said radient
    virtual void accum_partial_gradient(int i, Domain &x, Domain &grad, double step_size) const = 0;
    virtual Domain full_gradient(Domain &x) const = 0;

    virtual double operator() (const Domain &x) const = 0;
};

} // namespace Optimastic

#endif
