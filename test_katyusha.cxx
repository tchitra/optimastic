#include <iostream>
#include "random.hxx"
#include "function.hxx"
#include "katyusha.hxx"

using namespace SGD;

int main(void) {
    // Generate pnng
    std::cout << "Setting up random number generators\n";
    random_int<2> prng2;
    random_int<3> prng3;

    // Generate function
    std::cout << "Setting up quadratic test functions\n";
    Quadratic<2> q2;
    Quadratic<3> q3;

    // Generate Katyusha instance
    std::cout << "Setting up Katyusha instances\n";
    Katyusha<Quadratic<2> > k2(q2, 1.0, 1.0, 10, true, &prng2);
    Katyusha<Quadratic<3> > k3(q3, 1.0, 1.0, 10, true, &prng3);
    
    std::cout << "About to start optimization\n";
    for (int i=0; i<10; i++) { 
        k2.compute_single_window();
        k3.compute_single_window();

        printf("Window %i, 2d_min (%g), 3d_min (%g)\n", 
                i, k2.min(), k3.min());
    }
                
    return 0;
}
