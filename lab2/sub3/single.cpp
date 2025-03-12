#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <iomanip>

size_t N;
size_t threads;

void MatrixVectorProduct(const std::vector<double> &matrix, const std::vector<double> &vector,
                         std::vector<double> &resultVector, int lb, int ub) {
    for (int i = lb; i <= ub; ++i) {
        resultVector[i] = 0;
        for (int j = 0; j < N; ++j)
            resultVector[i] += matrix[i * N + j] * vector[j];
    }
}

void VectorSubtraction(const std::vector<double> &vector0, const std::vector<double> &vector1,
                       std::vector<double> &resultVector, int lb, int ub) {
    for (int i = lb; i <= ub; ++i)
        resultVector[i] = vector0[i] - vector1[i];
}

void ScalarVectorProduct(double scalar, const std::vector<double> &vector, std::vector<double> &resultVector,
                         int lb, int ub) {
    for (int i = lb; i <= ub; ++i)
        resultVector[i] = scalar * vector[i];
}

double squaredNorm(const std::vector<double> &vector, int lb, int ub) {
    double result{};
    for (int i = lb; i <= ub; ++i)
        result += vector[i] * vector[i];
    return result;
}

void Algorithm(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &X, double tau) {
    std::vector<double> bufferVector(N);
    double numerator{}, denominator{};
#pragma omp parallel num_threads(threads)
    {
        int nThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();
        int items_per_thread = N / nThreads;
        int lb = threadId * items_per_thread;
        int ub = (threadId == nThreads - 1) ? (N - 1) : (lb + items_per_thread - 1);
        double numBuf{}, denomBuf{};
        while (true) {
            MatrixVectorProduct(A, X, bufferVector, lb, ub);
            VectorSubtraction(bufferVector, b, bufferVector, lb, ub);
            numBuf = squaredNorm(bufferVector, lb, ub);
            denomBuf = squaredNorm(b, lb, ub);
#pragma omp single
            {
                numerator = 0;
                denominator = 0;
            }
#pragma omp atomic
                numerator += numBuf;
#pragma omp atomic
                denominator += denomBuf;
#pragma omp single
            {
                numerator = sqrt(numerator);
                denominator = sqrt(denominator);
            }
            if (numerator < 0.00001 * denominator)
                break;
            ScalarVectorProduct(tau, bufferVector, bufferVector, lb, ub);
            VectorSubtraction(X, bufferVector, X, lb, ub);
        }
    }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]);
    double tau = std::stod(argv[2]);
    threads = atoi(argv[3]);

    std::vector<double> A(N * N, 1);
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < N; ++i)
        A[i * N + i] = 2;

    const std::vector<double> b(N, N + 1);
    std::vector<double> X(N, 0);

    const auto start{std::chrono::steady_clock::now()};
    Algorithm(A, b, X, tau);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout << "Elapsed time: " << std::fixed << std::setprecision(5) << elapsed_seconds.count() << std::endl;
    return 0;
}
