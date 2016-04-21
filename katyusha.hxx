#ifndef __KATYUSHA_HXX__
#define __KATYUSHA_HXX__

#include <algorithm>

#ifdef __DEBUG
#include <iostream>
#endif

#include "random.hxx"
#include "ifunction.hxx"
#include "ioptimizer.hxx"

namespace Optimastic { 

template <typename Function>
class Katyusha : public IOptimizer<Function> {
    public:
        // typedefs (FIXME: The interface should provide these..)
        static const int Dimension = Function::Dimension;
        typedef typename Function::Domain Domain; // Should be some type of vector from eigen

        Katyusha(Function f, Domain initial_position, 
                double lipschitz_constant, double convexity_modulus, 
                int window_size, bool proximal, random_int<Dimension> *prng_ptr) 
            : _f(f)
            , _window_size(window_size)
            , _lipschitz_constant(lipschitz_constant)
            , _convexity_modulus(convexity_modulus)
            , _proximal(proximal)
            , _prng_ptr(prng_ptr)
        {
            // Set up constants
            _tau1 = std::min(0.5, sqrt(window_size * convexity_modulus / (3*lipschitz_constant)));
            _tau2 = 0.5;
            _alpha = 1./(3*_tau1*lipschitz_constant);

            // Compute the normalizing constant
            // N = (sum( (1-alpha*convexity_modulus)^j, 0, window_size-1))^-1
            // Use Horner's Rule to evaluate this
            _normalizer = 1; 
            double r = 1 + _alpha * _convexity_modulus;
            for (int i=0; i<window_size; i++) {
                _normalizer = r * (_normalizer+1); 
            }
            _normalizer = 1.0/_normalizer;
        }

        // compute_single_window returns the current step if successful, otherwise returns 0
        void compute_single_window();

        void run_optimizer(size_t k) { 
           auto niter_left = k;
           while (niter_left>0) { 
              compute_single_window();
#ifdef __DEBUG 
              std::cout << "Number of iterations left: " << niter_left << "\n";
#endif
           }
        }
        
        const Domain &argmin() const {
            return _last_mean;
        }

        const double min() const {
            return _f(_last_mean);
        }

    private: 
        const Function _f;

        // Katyusha state 
        // x is the current position of the iteration
        // y, z, z_prev are momentum variables
        // last_mean is the mean from the last minibatch
        Domain _x;
        Domain _y, _y_prev;
        Domain _z, _z_prev;
        Domain _last_mean; 

        // Constants
        size_t _size;
        size_t _window_size; // FIXME: Could be a smaller type
        size_t _max_iter;

        double _tau1, _tau2; 
        double _alpha;

        double _lipschitz_constant;
        double _convexity_modulus;
        double _normalizer;

        bool _proximal;
        size_t _current_total_steps

        // PRNG
        random_int<Dimension> *_prng_ptr;
};

template <typename Function>
void Katyusha<Function>::compute_single_window() { 
    // First update mean
    Domain full_grad = _f.full_gradient(_last_mean); 
    Domain accum_grad;  // For holding grad F(x) + grad_i F(x_proposed) - grad_i F(x)
    Domain accum_x;
    double  curr_weight = 1;

    for (int j=0; j<_window_size; j++) {
       int i = _prng_ptr->generate(); 

        // Update x to x[k+1]
       _x = _tau1 * _z + _tau2 * _last_mean + (1-_tau1-_tau2) * _y;

       // Generate diff; note that this simply involves  
       accum_grad = full_grad;
       _f.accum_partial_gradient(i, _x, accum_grad, 1.0); // + +grad_i(x)
       _f.accum_partial_gradient(i, _last_mean, accum_grad, -1.0); // =grad_i(last_mean)

       _z = _z - _alpha * accum_grad; // FIXME: This should be a proximal term in a full-optimization

       // Proximal update for Y:
       // If we want the true proximal / projected operator, we shift relative to the 
       // Lipschitz constant; otherwise, we take a constant  step size 
       _y = (_proximal) 
          ? _x - 1.0/(3.0*_lipschitz_constant) * accum_grad
          : _x - _tau1 * _alpha * accum_grad; // z[k+1]-z[k] = -alpha*accum_grad

       accum_x += curr_weight * _x;
       curr_weight *= (1+_alpha*_convexity_modulus);
       
       _current_total_steps++;
    }
    
    _last_mean = _normalizer * accum_x; 
    this->_current_step++; 
}

} // namespace SGD

#endif 
