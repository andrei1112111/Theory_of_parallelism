#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
// #include <boost/program_options.hpp>
#include "nvtx3/nvToolsExt.h"
#include <openacc.h>

void PrintMatrix(const double *grid, size_t n) {
#pragma acc update host(grid[0:n*n])
    for (size_t y = 0; y < n; ++y) {
        for (size_t x = 0; x < n; ++x) {
            std::cout << grid[y * n + x] << " ";
        }
        std::cout << std::endl;
    }
}

// int ProgramOptions(int argc, char **argv, double &epsilon, size_t &n, size_t &n_max_iterations) {
//     namespace po = boost::program_options;
//     po::options_description desc("Allowed flags");
//     desc.add_options()
//             ("help,h", "Show this text")
//             ("epsilon,e", "Epsilon")
//             ("size,n", "Matrix size")
//             ("steps,s", "Max steps");
//     po::variables_map vm;
//     po::store(po::parse_command_line(argc, argv, desc), vm);
//     po::notify(vm);

//     if (vm.count("help")) {
//         std::cout << desc << std::endl;
//         return 0;
//     }

//     epsilon = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 0.001;
//     n = (vm.count("size")) ? vm["size"].as<size_t>() : 10;
//     n_max_iterations = (vm.count("steps")) ? vm["steps"].as<size_t>() : 1000;

//     return 1;
// }

void InitializeGrid(double *grid, double *new_grid, size_t n, double val0, double val1, double val2, double val3) {
    auto const Interpolation = [](size_t x, size_t x0, size_t x1, double f_x0, double f_x1) {
        return f_x0 + ((f_x1 - f_x0) / static_cast<double>(x1 - x0)) * static_cast<double>(x - x0);
    };
    memset(grid, 0, n * n * sizeof(double));
    memset(new_grid, 0, n * n * sizeof(double));
    // Corner values initialization
    grid[0] = val0;              // 0----1
    grid[n - 1] = val1;          // |    |
    grid[n * n - 1] = val2;      // |    |
    grid[n * (n - 1)] = val3;    // 3----2
    new_grid[0] = val0;              // 0----1
    new_grid[n - 1] = val1;          // |    |
    new_grid[n * n - 1] = val2;      // |    |
    new_grid[n * (n - 1)] = val3;    // 3----2
    // Linear interpolation
#pragma acc enter data copyin(grid[0:n*n], new_grid[0:n*n])
#pragma acc parallel loop present(grid, new_grid)
    for (size_t i = 1; i < n - 1; ++i) {
        grid[i] = Interpolation(i, 0, n - 1, grid[0], grid[n - 1]); // 0-1
        grid[i * n + n - 1] = Interpolation(i, 0, n - 1, grid[n - 1], grid[n * n - 1]); // 1-2
        grid[n * (n - 1) + i] = Interpolation(i, 0, n - 1, grid[n * (n - 1)], grid[n * n - 1]); // 3-2
        grid[i * n] = Interpolation(i, 0, n - 1, new_grid[0], new_grid[n * (n - 1)]); // 0-3
        new_grid[i] = Interpolation(i, 0, n - 1, new_grid[0], new_grid[n - 1]); // 0-1
        new_grid[i * n + n - 1] = Interpolation(i, 0, n - 1, new_grid[n - 1], new_grid[n * n - 1]); // 1-2
        new_grid[n * (n - 1) + i] = Interpolation(i, 0, n - 1, new_grid[n * (n - 1)], new_grid[n * n - 1]); // 3-2
        new_grid[i * n] = Interpolation(i, 0, n - 1, new_grid[0], new_grid[n * (n - 1)]); // 0-3
    }
}

void Deallocate(double *grid, double *new_grid) {
#pragma acc exit data delete(grid, new_grid)
    delete[](grid);
    delete[](new_grid);
}

double CalculateNext(const double *grid, double *new_grid, size_t n) {
    double error{};
#pragma acc parallel loop reduction(max:error) present(grid, new_grid)
    for (size_t y = 1; y < n - 1; ++y) {
#pragma acc loop
        for (size_t x = 1; x < n - 1; ++x) {
            new_grid[y * n + x] = 0.2 * (grid[y * n + x]
                                         + grid[y * (n - 1) + x]
                                         + grid[y * n + x + 1]
                                         + grid[y * (n + 1) + x]
                                         + grid[y * n + x - 1]);
            error = fmax(error, fabs(grid[y * n + x] - new_grid[y * n + x]));
        }
    }
    return error;
}

void PrintMatrix(const std::vector<double> &matrix, size_t n) {
    for (size_t y = 0; y < n; ++y) {
        for (size_t x = 0; x < n; ++x) {
            std::cout << matrix[y * n + x] << " ";
        }
        std::cout << std::endl;
    }
}

void StupidSwap(double *grid, const double *new_grid, size_t n) {
#pragma acc parallel loop present(grid, new_grid)
    for (int y = 1; y < n - 1; ++y) {
#pragma acc loop
        for (int x = 1; x < n - 1; ++x) {
            grid[y * n + x] = new_grid[y * n + x];
        }
    }
}

int main(int argc, char **argv) {
    nvtxRangePushA("init");
    double epsilon = 0.000001;
    size_t n = 512;
    size_t n_max_iterations = 1000000;
    // if (!ProgramOptions(argc, argv, epsilon, n, n_max_iterations))
        // return 0;

    auto *grid = new double[n * n];
    auto *new_grid = new double[n * n];
    InitializeGrid(grid, new_grid, n, 10, 20, 30, 20);
    nvtxRangePop();

    size_t last_step{};
    double error = 1;
    auto const start = std::chrono::steady_clock::now();
    nvtxRangePushA("loop");
    for (size_t i{}; i < n_max_iterations && error > epsilon; ++i) {
        nvtxRangePushA("calc");
        error = CalculateNext(grid, new_grid, n);
        nvtxRangePop();
        nvtxRangePushA("swap");
        StupidSwap(grid, new_grid, n);
        nvtxRangePop();
        last_step = i;
    }
    nvtxRangePop();
    auto const end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds(end - start);
    if (n <= 13)
        PrintMatrix(grid, n);
    std::cout << last_step << "\n" << elapsed_seconds.count() << std::endl;
    Deallocate(grid, new_grid);
}