#ifndef __SVRG_HXX__
#define __SVRG_HXX__

#include "random.hxx"
#include "ifunction.hxx"
#include "ioptimizer.hxx"

namespace Optimastic { 
    template <typename Function>
    class SVRG : public IOptimizer<Function>  {
        public:
          // typedefs 
          static const int Dimension = Function::Dimension;
          typedef typename Function::Domain Domain;

          SVRG(Function f, Domain initial_condition,
              double step_size, double decay_offset,
              size_t mb_size, random_int<Dimension> *prng_ptr)
              : _current_min(initial_condition)
              , _step_size(step_size)
              , _decay(decay_offset)
              , _mb_size(mb_size)
              , _mb_nsteps(0)
              , _prng_ptr(prng_ptr)

          {
              this->_current_step = 1; // actual step, not zero indexed, since we divide by this

          }

          void run_single_batch() { 
              size_t steps_left = _mb_size;
              _mb_lru_gradient = this->_f.full_gradient(_current_min);

              // mb_accum will serve as w_t in Algortihm 1 from the paper
              // 
              // "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
              // Johnson, Zhang, NIPS, 2014
              //
              Domain mb_accum = _current_min;
              
              while (steps_left > 0) {
                  // Setup loop constants
                  double prefactor = -_step_size / (_decay + this->_current_step);
                  int i = _prng_ptr->generate(); 

                  // Compute partial gradient diff
                  Domain grad_diff;
                  for (int i=0; i<Dimension; i++) 
                      grad_diff[i] = 0;
                  
                  // SVRG performs the following iterate:
                  // 
                  // w[t] = w[t-1] - step * ( grad_f(w[t-1]) - grad_f[_current_min] + _mb_lru_gradient ) 
                  //
                  this->_f.accum_partial_gradient(i, mb_accum, mb_accum, prefactor); 
                  this->_f.accum_partial_gradient(i, _current_min, mb_accum, -prefactor);
                  mb_accum += prefactor * _mb_lru_gradient;

                  // Step update
                  this->_current_step++;
                  steps_left--;
              }
              _current_min = mb_accum;
              _mb_nsteps++;
          }

          void run_optimizer(size_t k) { 
              for (int i=0; i<k; i++) { 
                  run_single_batch();
              }
          }

          const Domain& argmin() const { 
              return _current_min;
          }

          const double min() const { 
              return this->_f(_current_min);
          }

          void print_step_state() const { 
              std::cout << "SVRG has complete a total of " << _mb_nsteps << " minibatch steps and " << this->_current_step << " total steps\n";
          }

        private:
          // N.B. No momentum, because variance accelerated SGD w/ momentum is Katyusha
          Domain _current_min;

          // Generic SGD parameters
          double _step_size;
          double _decay;

          // minibatch parameters
          Domain _mb_lru_gradient; 
          size_t _mb_size;
          size_t _mb_nsteps;

          random_int<Dimension> *_prng_ptr;
    };

} // namespace Optimastic

#endif
