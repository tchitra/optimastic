#ifndef __FUNCTION_HXX__
#define __FUNCTION_HXX__

#include "ifunction.hxx"

// Define a few basic functions 
namespace SGD { 

// Quadratic evaluates the gradient and partial
// gradient for the function
//
// f(x) = 1/2 * c0 x0^2 + c1 * x1^2 + ... + ck xk^2 + b
//
// where b is a translation
//
template <int n>
struct Quadratic : public IFunction<n> {
    typedef Matrix<double, n, 1> Domain;

    // FIXME: This initializer is likely slow
    Quadratic () 
    {
        for (int i=0; i<n; i++) {
            _coefficients[i] = 1.0;
            _shift[i]        = 0.0;
        }
    }

    Quadratic (Domain & coefficients, Domain & shift)
        : _coefficients(coefficients)
        , _shift(shift)
    {}

    Domain _full_gradient(Domain &x) { 
        Domain ret;
        for (int i=0; i<n; i++) { 
            ret[i] = _coefficients[i] * x[i] + _shift[i];
        }
        return ret;
    }

    double operator()(const Domain &x) {
        double ret = 0.0;
        for (int i=0; i<n; i++) { 
            ret = _coefficients[i] * x[i] * x[i] + _shift[i];
        }
        return ret;
    }

    // FIXME: Check and ensure that the compiler inlines this
    void _accum_partial_gradient(int i, Domain &x, Domain &grad, double step_size) { 
        grad[i] += step_size * _coefficients[i] * x[i] + _shift[i]; 
    }

    Domain _coefficients;
    Domain _shift;
};

}; // namespace SGD

#endif 
