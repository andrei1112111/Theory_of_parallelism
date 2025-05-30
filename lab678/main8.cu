#include <iostream>
#include <chrono>
#include <cuda.h>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#define BLOCK_SIZE 256

__global__ void InitializeEdges(double *grid, double *new_grid, size_t n,
                                double val0, double val1, double val2, double val3) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n - 2) return;
    ++i;

    auto interp = [](size_t x, size_t x0, size_t x1, double f0, double f1) {
        return f0 + ((f1 - f0) / (double)(x1 - x0)) * (double)(x - x0);
    };

    grid[i] = interp(i, 0, n - 1, val0, val1);
    grid[i * n + (n - 1)] = interp(i, 0, n - 1, val1, val2);
    grid[n * (n - 1) + i] = interp(i, 0, n - 1, val3, val2);
    grid[i * n] = interp(i, 0, n - 1, val0, val3);

    new_grid[i] = grid[i];
    new_grid[i * n + (n - 1)] = grid[i * n + (n - 1)];
    new_grid[n * (n - 1) + i] = grid[n * (n - 1) + i];
    new_grid[i * n] = grid[i * n];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        grid[0] = val0;
        grid[n - 1] = val1;
        grid[n * (n - 1)] = val3;
        grid[n * n - 1] = val2;

        new_grid[0] = val0;
        new_grid[n - 1] = val1;
        new_grid[n * (n - 1)] = val3;
        new_grid[n * n - 1] = val2;
    }
}

__global__ void UpdateKernel(const double *grid, double *new_grid, size_t n) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x > 0 && x < n - 1 && y > 0 && y < n - 1) {
        size_t idx = y * n + x;
        new_grid[idx] = 0.2 * (grid[idx] + grid[idx - 1] + grid[idx + 1]
                             + grid[idx - n] + grid[idx + n]);
    }
}

__global__ void MaxErrorBlockReduce(const double *a, const double *b, double *block_max_errors, size_t size) {
    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    double thread_error = 0.0;

    if (idx < size) {
        thread_error = fabs(a[idx] - b[idx]);
    }

    double block_max = BlockReduce(temp_storage).Reduce(thread_error, cub::Max());
    if (threadIdx.x == 0) {
        block_max_errors[blockIdx.x] = block_max;
    }
}

void PrintMatrix(const double *matrix, size_t n) {
    for (size_t y = 0; y < n; ++y) {
        for (size_t x = 0; x < n; ++x) {
            std::cout << matrix[y * n + x] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    constexpr size_t n = 1024;
    constexpr size_t size = n * n;
    constexpr double epsilon = 0.000001;
    constexpr size_t max_iter = 1000000;

    double *d_grid, *d_new_grid;
    cudaMalloc(&d_grid, size * sizeof(double));
    cudaMalloc(&d_new_grid, size * sizeof(double));

    cudaMemset(d_grid, 0, size * sizeof(double));
    cudaMemset(d_new_grid, 0, size * sizeof(double));

    dim3 block1d(BLOCK_SIZE);
    dim3 grid1d((n - 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    InitializeEdges<<<grid1d, block1d>>>(d_grid, d_new_grid, n, 10.0, 20.0, 30.0, 20.0);
    cudaDeviceSynchronize();

    dim3 block2d(16, 16);
    dim3 grid2d((n + 15) / 16, (n + 15) / 16);

    size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *d_block_errors;
    cudaMalloc(&d_block_errors, num_blocks * sizeof(double));

    double *h_block_errors = new double[num_blocks];
    double max_error = 1.0;
    size_t iter = 0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_created = false;

    auto start = std::chrono::high_resolution_clock::now();

    const int error_check_interval = 1000;

    nvtxRangePushA("loop");
    while (max_error >= epsilon && iter < max_iter) {
        if (iter % error_check_interval == 0) {
            nvtxRangePushA("calc");
            if (!graph_created) {
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
                for (int k = 0; k < error_check_interval; ++k) {
                    UpdateKernel<<<grid2d, block2d, 0, stream>>>(d_grid, d_new_grid, n);
                    std::swap(d_grid, d_new_grid);
                }
                MaxErrorBlockReduce<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
                    d_grid, d_new_grid, d_block_errors, size);
                cudaStreamEndCapture(stream, &graph);
                cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
                graph_created = true;
            }

            cudaGraphLaunch(graph_exec, stream);
            cudaStreamSynchronize(stream);

            cudaMemcpy(h_block_errors, d_block_errors, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
            nvtxRangePop();
            nvtxRangePushA("error");
            max_error = 0.0;
            for (size_t i = 0; i < num_blocks; ++i) {
                if (h_block_errors[i] > max_error)
                    max_error = h_block_errors[i];
            }
            nvtxRangePop();
            iter += error_check_interval;
        } else {
            nvtxRangePushA("calc");
            UpdateKernel<<<grid2d, block2d>>>(d_grid, d_new_grid, n);
            nvtxRangePop(); 
            std::swap(d_grid, d_new_grid);
            ++iter;
        }
    }
    nvtxRangePop();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;

    std::cout << "Finished in " << iter << " steps.\n";
    std::cout << "Time elapsed: " << time.count() << " s\n";
    std::cout << "err " << max_error << "\n";

    if (n <= 13) {
        double *h_grid = new double[size]; 
        cudaMemcpy(h_grid, d_grid, size * sizeof(double), cudaMemcpyDeviceToHost);
        PrintMatrix(h_grid, n);
        delete[] h_grid;
    }

    cudaFree(d_grid);
    cudaFree(d_new_grid);
    cudaFree(d_block_errors);
    delete[] h_block_errors;

    if (graph_created) {
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);

    return 0;
}
