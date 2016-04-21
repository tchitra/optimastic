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
#include "ioptimizer.hxx"
#include "function.hxx"
#include "katyusha.hxx"
#include "sgd.hxx"
#include "random.hxx"

#define SMALL_DIMENSION 2
#define BIG_DIMENSION   10

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

    IOptimizer<Quadratic<SMALL_DIMENSION> > *opt_small; 
    IOptimizer<Quadratic<BIG_DIMENSION> >   *opt_big;

    if (name == "Katyusha") {  
        std::cout << "Setting up Katyusha instances\n";
        Katyusha<Quadratic<SMALL_DIMENSION> >  kSmall (qSmall, initialSmall, 10.0, 1.0, 10, false, &prngSmall);
        Katyusha<Quadratic<BIG_DIMENSION> >    kBig   (qBig, initialBig, 10.0, 1.0, 10, false, &prngBig);

        opt_small = &kSmall;
        opt_big   = &kBig;
    } else if ( name == "SGD") {
        std::cout << "Setting up SGD instances\n";
        SGD<Quadratic<SMALL_DIMENSION> >  sgdSmall (qSmall, initialSmall, 1.0, 1.0, &prngSmall);
        SGD<Quadratic<BIG_DIMENSION> >    sgdBig   (qBig,   initialBig,   1.0, 1.0, &prngBig);

        opt_small = &sgdSmall;
        opt_big   = &sgdBig;
    } else {
        std::cout << "Method name " << name << "not found, aborting";
        abort();
    }

    std::cout << "About to start optimization\n";
    for (int i=0; i<10; i++) { 
        opt_small->run_optimizer(10000);
        opt_big->run_optimizer(10000);

        printf("Window %i, small min (%g), big min (%g), small norm (%g), big norm (%g)\n", 
                i, opt_small->min(), opt_big->min(), 
                opt_small->argmin().norm(), opt_big->argmin().norm());
    }
}



int main(int argc, char **argv) { 
    NameVec methods;
    size_t max_dim, seed;
    parse_args(argc, argv, methods, max_dim, seed);

    if ( methods.size() == 0 ) {
        printf("%s -m [method] -d [max_dim] -s [seed]\n", argv[0]);
        exit(1); 
    }

    for (auto name : methods) { 
        run_tests(name);
    }

    return 0;
}
