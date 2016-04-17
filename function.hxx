#ifndef __FUNCTION_HXX__
#define __FUNCTION_HXX__

#include "ifunction.hxx"

// Define a few basic functions 
namespace Optimastic { 

// Quadratic evaluates the gradient and partial
// gradient for the function
//
// f(x) = 1/2 * c0 x0^2 + c1 * x1^2 + ... + ck xk^2 + b
//
// where b is a translation
//
template <int n>
struct Quadratic : public IFunction<n> {
    typedef typename IFunction<n>::Domain Domain;

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

    virtual Domain full_gradient(Domain &x) const { 
        Domain ret;
        for (int i=0; i<n; i++) { 
            ret[i] = _coefficients[i] * x[i] + _shift[i];
        }
        return ret;
    }

    virtual double operator()(const Domain &x) const {
        double ret = 0.0;
        for (int i=0; i<n; i++) { 
            ret = _coefficients[i] * x[i] * x[i] + _shift[i];
        }
        return ret;
    }

    // FIXME: Check and ensure that the compiler inlines this
    virtual void accum_partial_gradient(int i, Domain &x, Domain &grad, double step_size) const { 
        grad[i] += step_size * _coefficients[i] * x[i] + _shift[i]; 
    }

    Domain _coefficients;
    Domain _shift;
};

}; // namespace Optimastic

#endif 
