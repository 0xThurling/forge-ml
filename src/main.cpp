#include "./nn/engine.hpp"
#include <iostream>

int main() {
    forgeml::Value x{2.0};
    forgeml::Value y{-3.0};
    forgeml::Value z{10.0};

    forgeml::Value f = x * y + z;

    f.backward();

    std::cout << "f.data = " << f.data() << "\n"; 
    std::cout << "df/dx = " << x.grad() << "\n"; 
    std::cout << "df/dy = " << y.grad() << "\n"; 
    std::cout << "df/dz = " << z.grad() << "\n"; 
}
