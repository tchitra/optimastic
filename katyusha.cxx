#include "katyusha.hxx"

namespace SGD { 

template <typename Function, typename T>
void Katyusha::compute_single_window() { 
    // First update mean
    KVector full_grad = _f.full_gradient(_last_mean); 
    KVector accum_grad;
    KVector accum_x;
    double  curr_weight = 1;

    for (int j=0; j<_window_size; j++) {
       int i = gen_random_idx(_f.size());

        // Update x to x[k+1]
       _x = _tau1 * _z + _tau2 * _last_mean + (1-_tau1-_tau2) * _y;

       // Generate diff 
       accum_grad = full_grad;
       _f.partial_gradient(i, _x, accum_grad, 1.0); // + +grad_i(x)
       _f.partial_gradient(i, _last_mean, accum_grad, -1.0); // =grad_i(last_mean)

       _z = _z - _alpha * accum_grad; // FIXME: This should be a proximal term in a full-optimization

       // Proximal update for Y:
       // If we want the true proximal / projected operator, we shift relative to the 
       // Lipschitz constant; otherwise, we take a constant  step size 
       _y = (proximal) 
          ? _x - 1.0/(3.0*_lipschitz_constant) * accum_grad
          : _x - _tau1 * _alpha * accum_grad; // z[k+1]-z[k] = -alpha*accum_grad

       accum_x += curr_weight * _x;
       curr_weight *= 1+_alpha*_convexity_modulus;
    }
    
    _last_mean = _normalizer * accum_x; 
}

} // namespace SGD
