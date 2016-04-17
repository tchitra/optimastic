#ifndef __IOPTIMIZE_HXX__
#define __IOPTIMIZE_HXX__

//
// IOptimize is an interface for stochastic optimizers 
// 
// Design principles:
// a. The interface must be state agnostic
// b. The function type, provided as a template argument, must provide
//    everything --- constraints, domain, full gradient, partial gradient 
// 
// N.B. 
// Principle (b) might be annoying to deal with when we have a Python interface...
//

namespace Optimastic { 

template <typename Function>
class IOptimizer { 
    public:
        // Constants / Constraints from template arg
        typedef typename Function::Domain Domain;
        static const int Dimension = Function::Dimension;

        // Accessors
        virtual const Domain& argmin() const = 0;
        virtual const double  min()    const = 0;

        // Run optimizer for k iterations
        virtual void run_optimizer(size_t k) = 0;

    private:
        // FIXME: Only store a pointer/reference, eventually
        const Function _f; 
};

} // namespace Optimastic

#endif
