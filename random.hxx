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
// 3. Seed randomization
//

#define SEED 245201

#include <random>

namespace Optimastic { 
    template <int n>
    struct random_int {
        random_int<n>() {
            // N.B. std::uniform_int_distribution *includes* the endpoints
            dist = new std::uniform_int_distribution<int>(0,n-1);  
        }

        ~random_int<n>() { 
            delete dist;
        }

        // For now, we'll make this noncopyable for now
        random_int<n>(const random_int<n> &) = delete;


        int generate() { 
            return (*dist)(generator);
        }

        std::default_random_engine generator; 
        std::uniform_int_distribution<int>* dist;
    };
}

#endif // __RANDOM_HXX__
