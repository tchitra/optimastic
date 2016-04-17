#include <Eigen/Dense>
#include <algorithm>

template <typename Function, int n>
class Katyusha {
    public:
        typedef Matrix<double, n, 1> KVector;

        Katyusha(Function f, 
                double lipschitz_constant, double convexity_modulus 
                int window_size, bool proximal) 
            : _f(f)
            , _window_size(window_size)
            , _lipschitz_constant(lipschitz_constant)
            , _convexity_modulus(convexity_modulus)
            , _proximal(proximal)
        {
            _x.resize(n);
            _y.resize(n);
            _z.resize(n);
            _last_mean.resize(n);

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
        }

        virtual ~Katyusha(); // We might have derived classes if we have threads

        // compute_single_window returns the current step if successful, otherwise returns 0
        void compute_single_window();

        inline KVector get_opt_value() {
            return _last_mean;
        }

    private: 
        Function _f;

        // Katyusha state 
        // x is the current position of the iteration
        // y, z, z_prev are momentum variables
        // last_mean is the mean from the last minibatch
        //
        // FIXME: Separate this out into a struct
        KVector _x;
        KVector _y, _y_prev;
        KVector _z, _z_prev;
        KVector _last_mean; 

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
};
