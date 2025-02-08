#include <iostream>
#include <cmath>


#ifdef USE_DOUBLE
#define TYPE double
#else
#define TYPE float
#endif


#define data_size 10000000
#define pi (M_PI * 2 / data_size)


int main() {
    std::unique_ptr<TYPE[]> data(new TYPE[data_size]);
    for (unsigned i = 0; i < data_size; ++i) {
        data[i] = sin(i * pi);
    }

    TYPE sum = 0;
    for (unsigned i = 0; i < data_size; ++i) {
        sum += data[i];
    }
    std::cout << sum << std::endl;

    return 0;
}
