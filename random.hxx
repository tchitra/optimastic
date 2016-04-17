#ifndef __RANDOM_HXX__
#define __RANDOM_HXX__

//
// This header defines a series of wrappers for the standard random number generators
// that are standardized to our needs and let prngs be created as compile-time instead
// of at run-time.
//
// FIXME: 
// 1. Use random123?
// 2. Thread-safety?
//

#include <random>
#define SEED 245201

namespace SGD { 
    template <int n>
    struct random_int {
        int generate() { 
            return dist(generator);
        }

        std::default_random_engine generator (SEED); 
        std::uniform_int_distribution dist (0, n);
    };
}

#endif // __RANDOM_HXX__
