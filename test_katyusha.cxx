#include <iostream>
#include <cmath>
#include "random.hxx"
#include "function.hxx"
#include "katyusha.hxx"

#define SMALL_DIMENSION 10 
#define BIG_DIMENSION   15

using namespace Optimastic;

int main(void) {
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

    // Generate Katyusha instance
    std::cout << "Setting up Katyusha instances\n";
    Katyusha<Quadratic<SMALL_DIMENSION> >  kSmall (qSmall, initialSmall, 10.0, 1.0, 10, false, &prngSmall);
    Katyusha<Quadratic<BIG_DIMENSION> >    kBig   (qBig, initialBig, 10.0, 1.0, 10, false, &prngBig);
    
    std::cout << "About to start optimization\n";
    for (int i=0; i<10000; i++) { 
        kSmall.compute_single_window();
        kBig.compute_single_window();

        if (i % 1000 == 0) { 
            printf("Window %i, small min (%g), big min (%g), small norm (%g), big norm (%g)\n", 
                    i, kSmall.min(), kBig.min(), kSmall.argmin().norm(), kBig.argmin().norm());
        }
    }
                
    return 0;
}
