#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <boost/program_options.hpp>
#include "nvtx3/nvToolsExt.h"
#include <openacc.h>
#include <cublas_v2.h>

void PrintMatrix(const double *grid, size_t n) {
#pragma acc update host(grid[0:n*n])
    for (size_t y = 0; y < n; ++y) {
        for (size_t x = 0; x < n; ++x) {
            std::cout << grid[y * n + x] << " ";
        }
        std::cout << std::endl;
    }
}

int ProgramOptions(int argc, char **argv, double &epsilon, size_t &n, size_t &n_max_iterations) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed flags");
    desc.add_options()
            ("help,h", "Show this text")
            ("epsilon,e", "Epsilon")
            ("size,n", "Matrix size")
            ("steps,s", "Max steps");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    epsilon = (vm.count("epsilon")) ? vm["epsilon"].as<double>() : 0.001;
    n = (vm.count("size")) ? vm["size"].as<size_t>() : 10;
    n_max_iterations = (vm.count("steps")) ? vm["steps"].as<size_t>() : 1000;

    return 1;
}

void InitializeGrid(double *grid, double *new_grid, size_t n, double val0, double val1, double val2,
                    double val3) {
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

double CalculateNext(double *grid, double *new_grid, size_t n, cublasHandle_t handle) {
    auto *error_grid = new double[n - 2];
#pragma acc enter data create(error_grid[0:n-2])
#pragma acc parallel loop async
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            new_grid[i * n + j] = 0.2 * (grid[i * n + j]
                                         + grid[i * (n - 1) + j]
                                         + grid[i * n + j + 1]
                                         + grid[i * (n + 1) + j]
                                         + grid[i * n + j - 1]);
            error_grid[i - 1] = fmax(error_grid[i - 1], fabs(grid[i * n + j] - new_grid[i * n + j]));
        }
    }
#pragma acc update host(new_grid[:n * n]) async
#pragma acc wait
    int max_err_id = 0;
#pragma acc host_data use_device(error_grid)
    {
        nvtxRangePushA("cublas");
        cublasIdamax(handle, n - 2, error_grid, 1, &max_err_id);
        nvtxRangePop();
    }
    double error = error_grid[max_err_id - 1];
#pragma acc exit data delete (error_grid[0:n-2])
    delete[](error_grid);
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

int main(int argc, char **argv) {
    nvtxRangePushA("init");
    double epsilon{};
    size_t n{};
    size_t n_max_iterations{};
    if (!ProgramOptions(argc, argv, epsilon, n, n_max_iterations))
        return 0;

    auto *grid = new double[n * n];
    auto *new_grid = new double[n * n];
    InitializeGrid(grid, new_grid, n, 10, 20, 30, 20);
    nvtxRangePop();

    size_t last_step{};
    double error = 1;
    auto const start = std::chrono::steady_clock::now();
    cublasHandle_t handle;
    cublasCreate(&handle);
    nvtxRangePushA("loop");
    for (size_t i{}; i < n_max_iterations && error > epsilon; ++i) {
        nvtxRangePushA("calc");
        error = CalculateNext(grid, new_grid, n, handle);
        nvtxRangePop();
        nvtxRangePushA("swap");
        std::swap(grid, new_grid);
        nvtxRangePop();
        last_step = i;
    }
    nvtxRangePop();
    cublasDestroy(handle);
    auto const end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds(end - start);
    if (n <= 13)
        PrintMatrix(grid, n);
    std::cout << last_step << "\n" << elapsed_seconds.count() << std::endl;
    Deallocate(grid, new_grid);
}
