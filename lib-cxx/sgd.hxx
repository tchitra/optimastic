#ifndef __SGD_HXX__
#define __SGD_HXX__

#include "random.hxx"
#include "ifunction.hxx"
#include "ioptimizer.hxx"

namespace Optimastic { 
    template <typename Function>
    class SGD : public IOptimizer<Function>  {
        public:
          // typedefs 
          static const int Dimension = Function::Dimension;
          typedef typename Function::Domain Domain;

          SGD(Function f, Domain initial_condition,
              double step_size, double decay_offset,
              random_int<Dimension> *prng_ptr,
              double friction_coefficient = 0.0, bool use_momentum = false)
              : _current_min(initial_condition)
              , _use_momentum(use_momentum)
              , _step_size(step_size)
              , _friction_coefficient(friction_coefficient)
              , _prng_ptr(prng_ptr)
          {
              this->_current_step = 1; // actual step, not zero indexed, since we divide by this

              if (use_momentum) {
                  // FIXME: Figure out how the correct default initialization should work for Eigen
                  for (int i=0; i<Dimension; i++) { 
                     _current_velocity[i] = 0;
                  }
              }
          }

          void run_optimizer(size_t k) { 
              size_t steps_left = k;
              while (steps_left > 0) {
                  // Setup loop constants
                  double prefactor = -_step_size / (_decay + this->_current_step);
                  int i = _prng_ptr->generate(); 

                  // Accumulate velocity
                  _current_velocity *= _friction_coefficient;
                  this->_f.accum_partial_gradient(i, _current_min, _current_velocity, prefactor); 
                  
                  // Update argmin
                  _current_min += _current_velocity;
                  
                  // Step update
                  this->_current_step++;
                  steps_left--;
              }
          }

          const Domain& argmin() const { 
              return _current_min;
          }

          const double min() const { 
              return this->_f(_current_min);
          }

          void print_step_state() const { 
              std::cout << "SGD is currently at iteration " << this->_current_step << "\n"; 
          }

        private:
          Domain _current_min;
          Domain _current_velocity;

          bool   _use_momentum;

          double _step_size;
          double _friction_coefficient; // N.B. The stochastic optimization community calls this "momentum" 
          double _learning_rate;
          double _decay;


          random_int<Dimension> *_prng_ptr;
    };

} // namespace Optimastic

#endif
