// C includes
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <getopt.h>

// C++ includes
#include <iostream>
#include <string>
#include <vector>

// Optimastic includes
#include "function.hxx"

using namespace Optimastic;

typedef std::vector<std::string> NameVec;

void parse_args(int argc, char **argv, NameVec &names, size_t &max_dim, size_t &seed) {
    // Parse options
    extern char *optarg;
    int c;

    while (true) { 
        static struct option long_options [] = 
        {
            { "method" , required_argument, 0, 'm' },
            { "max_dim", optional_argument, 0, 'd' },
            { "seed"   , optional_argument, 0, 's' }
        };

        int option_index = 0;
        c = getopt_long (argc, argv, "m:d:s:",
                long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) { 
            case 'm':
                std::cout << "Adding Method: " << optarg << "\n";
                names.push_back(optarg);
                break;
            case 'd':
                std::cout << "Max Dimension: " << optarg << "\n";
                max_dim = atoi(optarg);
                break;
            case 's':
                std::cout << "Seed: " << optarg << "\n";
                seed = atoi(optarg);
                break;
            case '?':
                break;
            default:
                std::cout << "Parsing failed, aborting\n";
                abort();
        }
} 

void run_tests(std::string &name) { 
    // FIXME: 
    // 1. This obviously will need to be more comprehensive in the future
    // 2. We should modularize this once we have a stable set of tests
    // Precompute sqrt factors
    
    const double sqrtSmall = sqrt(SMALL_DIMENSION);
    const double sqrtBig   = sqrt(BIG_DIMENSION);

    // Generate pnng
    std::cout << "Setting up random number generators\n";
    std::normal_distribution<>  norm_dist;
    random_int<SMALL_DIMENSION> prngSmall;
    random_int<BIG_DIMENSION>   prngBig;

    // Generate function
    std::cout << "Setting up quadratic test functions\n";
    Quadratic<SMALL_DIMENSION>::Domain coefSmall, shiftSmall; 
    Quadratic<BIG_DIMENSION>::Domain   coefBig, shiftBig;

    // N.B. Separating loops in case we need to print stuff
    for (int i=0; i<SMALL_DIMENSION; i++) { 
        coefSmall[i] = 1.0; // abs(norm_dist(prngSmall.generator));
        shiftSmall[i] = 1.0;
    }

    for (int i=0; i<BIG_DIMENSION; i++) { 
        coefBig[i] = 1.0;// abs(norm_dist(prngBig.generator));
        shiftBig[i] = 1.0;
    }

    Quadratic<SMALL_DIMENSION> qSmall; // (coefSmall, shiftSmall);
    Quadratic<BIG_DIMENSION>   qBig; // (coefBig, shiftBig);

    // Generate initial conditions
    Quadratic<SMALL_DIMENSION>::Domain initialSmall;
    Quadratic<BIG_DIMENSION>::Domain   initialBig;

    // Generate initial conditions
    std::cout << "initial condition for quadratic<2>: ";
    for (int i=0; i<SMALL_DIMENSION; i++) {
        initialSmall[i] = sqrtSmall * norm_dist(prngSmall.generator);
        std::cout << initialSmall[i] << "\t";
    }
    
    std::cout << "\ninitial condition for quadratic<3>: ";    
    for (int i=0; i<BIG_DIMENSION; i++) {
        initialBig[i] = sqrtBig * norm_dist(prngBig.generator);
        std::cout << initialBig[i] << "\t";
    }
    std::cout << "\n";

    IOptimizer *opt_small, *opt_big;

    switch (name) {
        case "Katyusha":  
            std::cout << "Setting up Katyusha instances\n";
            Katyusha<Quadratic<SMALL_DIMENSION> >  kSmall (qSmall, initialSmall, 10.0, 1.0, 10, false, &prngSmall);
            Katyusha<Quadratic<BIG_DIMENSION> >    kBig   (qBig, initialBig, 10.0, 1.0, 10, false, &prngBig);
            opt_small = &kSmall;
            opt_big   = &kBig;
            break;
        case "SGD":
           std::cout << "Setting up SGD instances\n";
           // FIXME
           break;
        default: 
           std::cout << "Method name " << name << "not found, aborting";
           abort()
    }

    std::cout << "About to start optimization\n";
    for (int i=0; i<10000; i++) { 
        opt_small->compute_single_window();
        opt_big->compute_single_window();

        if (i % 1000 == 0) { 
            printf("Window %i, small min (%g), big min (%g), small norm (%g), big norm (%g)\n", 
                    i, opt_small->min(), opt_big->min(), 
                    opt_small->argmin().norm(), opt_big->argmin().norm());
        }
    }
}



int main(int argc, char **argv) { 
    NameVec methods;
    size_t max_dim, seed;
    parse_args(argc, argv, names, max_dim, seed);

    for (auto name : names) { 
        run_tests(name);
    }

    return 0;
}
