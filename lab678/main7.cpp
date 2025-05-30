#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <nvtx3/nvToolsExt.h>
// #include <boost/program_options.hpp>
#include <cublas_v2.h>
#include <openacc.h>


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

    grid[0] = val0;
    grid[n - 1] = val1;
    grid[n * n - 1] = val2;
    grid[n * (n - 1)] = val3;
    new_grid[0] = val0;
    new_grid[n - 1] = val1;
    new_grid[n * n - 1] = val2;
    new_grid[n * (n - 1)] = val3;

#pragma acc enter data copyin(grid[0:n*n], new_grid[0:n*n])
#pragma acc parallel loop present(grid, new_grid)
    for (size_t i = 1; i < n - 1; ++i) {
        grid[i] = Interpolation(i, 0, n - 1, grid[0], grid[n - 1]);
        grid[i * n + n - 1] = Interpolation(i, 0, n - 1, grid[n - 1], grid[n * n - 1]);
        grid[n * (n - 1) + i] = Interpolation(i, 0, n - 1, grid[n * (n - 1)], grid[n * n - 1]);
        grid[i * n] = Interpolation(i, 0, n - 1, grid[0], grid[n * (n - 1)]);

        new_grid[i] = grid[i];
        new_grid[i * n + n - 1] = grid[i * n + n - 1];
        new_grid[n * (n - 1) + i] = grid[n * (n - 1) + i];
        new_grid[i * n] = grid[i * n];
    }
}

void Deallocate(double *grid, double *new_grid, double *diff, size_t n) {
#pragma acc exit data delete(grid[0:0], new_grid[0:0]) // фикс для intel/nvidia компиляторов
#pragma acc exit data delete(diff[0:n])
    delete[] grid;
    delete[] new_grid;
    delete[] diff;
}

void UpdateGrid(const double *grid, double *new_grid, size_t n) {
#pragma acc parallel loop collapse(2) present(grid, new_grid)
    for (size_t y = 1; y < n - 1; ++y) {
        for (size_t x = 1; x < n - 1; ++x) {
            size_t idx = y * n + x;
            new_grid[idx] = 0.2 * (
                grid[idx] +
                grid[idx - n] +
                grid[idx + 1] +
                grid[idx + n] +
                grid[idx - 1]
            );
        }
    }
}

double ComputeError(const double *grid, const double *new_grid, double *diff, size_t n, cublasHandle_t handle) {
    size_t size = n * n;

    // 1. Вычисляем разности: diff[i] = fabs(grid[i] - new_grid[i])
    #pragma acc parallel loop present(grid[0:size], new_grid[0:size], diff[0:size])
    for (size_t i = 0; i < size; ++i) {
        diff[i] = fabs(grid[i] - new_grid[i]);
    }

    int max_index = -1;
    double max_val = 0.0;

    // 2. Получаем device pointer и вызываем cuBLAS Idamax
    #pragma acc host_data use_device(diff)
    {
        cublasIdamax(handle, size, diff, 1, &max_index);
        cublasGetVector(1, sizeof(double), diff + (max_index - 1), 1, &max_val, 1);
    }

    return max_val;
}

void PrintMatrix(const double *matrix, size_t n) {
#pragma acc update host(matrix[0:n*n])
    for (size_t y = 0; y < n; ++y) {
        for (size_t x = 0; x < n; ++x) {
            std::cout << matrix[y * n + x] << " ";
        }
        std::cout << std::endl;
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
    auto *diff = new double[n * n]; // для cuBLAS

    InitializeGrid(grid, new_grid, n, 10, 20, 30, 20);
    #pragma acc enter data copyin(diff[0:n*n])

    // cuBLAS init
    cublasHandle_t handle;
    cublasCreate(&handle);

    nvtxRangePop(); // end init

    size_t last_step = 0;
    double error = 1.0;

    auto const start = std::chrono::steady_clock::now();
    nvtxRangePushA("loop");
    for (size_t i = 0; i < n_max_iterations && error > epsilon; ++i) {
        nvtxRangePushA("calc");
        UpdateGrid(grid, new_grid, n);
        nvtxRangePop();

        nvtxRangePushA("error");
        if (i % 1000 == 0) {
            error = ComputeError(grid, new_grid, diff, n, handle);
        }
        nvtxRangePop();

        nvtxRangePushA("swap");
        std::swap(grid, new_grid);
        nvtxRangePop();

        last_step = i;
    }
    nvtxRangePop(); // end loop

    auto const end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    if (n <= 13)
        PrintMatrix(grid, n);

    std::cout << "Steps: " << last_step << "\n";
    std::cout << "Time : " << elapsed_seconds.count() << "s\n";
    std::cout << "err " << error << "\n";

    cublasDestroy(handle);
    Deallocate(grid, new_grid, diff, n);
}

// nvc++ -acc -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/targets/x86_64-linux/include -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib -lcublas -o main7 main7.cpp

// ./main7
