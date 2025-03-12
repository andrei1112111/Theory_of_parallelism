#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

size_t N;
size_t threads;

void MatrixVectorProduct(const std::vector<double> &matrix, const std::vector<double> &vector,
                         std::vector<double> &resultVector) {
#if defined(SCHEDULE_DYNAMIC)
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#elif defined(SCHEDULE_GUIDED)
#pragma omp parallel for num_threads(threads) schedule(guided)
#elif defined(SCHEDULE_AUTO)
#pragma omp parallel for num_threads(threads) schedule(auto)
#else
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i = 0; i < N; ++i) {
        resultVector[i] = 0;
        for (int j = 0; j < N; ++j)
            resultVector[i] += matrix[i * N + j] * vector[j];
    }
}

void VectorSubtraction(const std::vector<double> &vector0, const std::vector<double> &vector1,
                       std::vector<double> &resultVector) {
#if defined(SCHEDULE_DYNAMIC)
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#elif defined(SCHEDULE_GUIDED)
#pragma omp parallel for num_threads(threads) schedule(guided)
#elif defined(SCHEDULE_AUTO)
#pragma omp parallel for num_threads(threads) schedule(auto)
#else
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i = 0; i < N; ++i)
        resultVector[i] = vector0[i] - vector1[i];
}

void ScalarVectorProduct(double scalar, const std::vector<double> &vector, std::vector<double> &resultVector) {
    for (int i = 0; i < N; ++i)
        resultVector[i] = scalar * vector[i];
}

double Norm(const std::vector<double> &vector) {
    double norm{};
#if defined(SCHEDULE_DYNAMIC)
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#elif defined(SCHEDULE_GUIDED)
#pragma omp parallel for num_threads(threads) schedule(guided)
#elif defined(SCHEDULE_AUTO)
#pragma omp parallel for num_threads(threads) schedule(auto)
#else
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i = 0; i < N; ++i)
        norm += vector[i] * vector[i];
    return sqrt(norm);
}

void Algorithm(const std::vector<double> &A, const std::vector<double> &b, std::vector<double> &X, double tau) {
    std::vector<double> bufferVector(N);
    while (true) {
        MatrixVectorProduct(A, X, bufferVector);
        VectorSubtraction(bufferVector, b, bufferVector);
        if (Norm(bufferVector) < 0.00001 * Norm(b))
            break;
        ScalarVectorProduct(tau, bufferVector, bufferVector);
        VectorSubtraction(X, bufferVector, X);
    }
}

int main(int argc, char **argv) {
    N = atoi(argv[1]);
    double tau = std::stod(argv[2]);
    threads = atoi(argv[3]);

    std::vector<double> A(N * N, 1);
#if defined(SCHEDULE_DYNAMIC)
#pragma omp parallel for num_threads(threads) schedule(dynamic)
#elif defined(SCHEDULE_GUIDED)
#pragma omp parallel for num_threads(threads) schedule(guided)
#elif defined(SCHEDULE_AUTO)
#pragma omp parallel for num_threads(threads) schedule(auto)
#else
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i = 0; i < N; ++i)
        A[i * N + i] = 2;

    const std::vector<double> b(N, N + 1);
    std::vector<double> X(N, 0);

    const auto start{std::chrono::steady_clock::now()};
    Algorithm(A, b, X, tau);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    std::cout << "Elapsed time: " << elapsed_seconds.count()
              << std::endl;
    return 0;
}
