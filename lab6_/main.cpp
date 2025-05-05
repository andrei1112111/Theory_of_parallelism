#include <iostream>
#include <vector>
#include <valarray>
#include <numeric>

void PrintMatrix(const std::vector<double> &matrix, size_t n) {
    for (size_t y = 0; y < n; ++y) {
        for (size_t x = 0; x < n; ++x) {
            std::cout << matrix[y * n + x] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::accumulate(matrix.begin(), matrix.end(), 0.) << std::endl;
}

double GetA(size_t row, size_t col, size_t n) {
    if (row == col) return -4;
    if (row + 1 == col) return 1;
    if (row - 1 == col) return 1;
    if (row + n == col) return 1;
    if (row - n == col) return 1;
    return 0;
}

void InitMatrix(std::vector<double> &matrix, size_t n) {
    auto n_squared = n * n;
    for (size_t y{}; y < n_squared; ++y)
        for (size_t x{}; x < n_squared; ++x)
            matrix[y * n_squared + x] = GetA(y, x, n);
}

double Interpolation(size_t x, size_t x0, size_t x1, double f_x0, double f_x1) {
    return f_x0 + ((f_x1 - f_x0) / static_cast<double>(x1 - x0)) * static_cast<double >(x - x0);
}

void InitNetwork(std::vector<double> &network, size_t n, double val0, double val1, double val2, double val3) {
    // Corner values initialization
    network[0] = val0;              // 0----1
    network[n - 1] = val1;          // |    |
    network[n * n - 1] = val2;      // |    |
    network[n * (n - 1)] = val3;    // 3----2
    // Linear interpolation
    for (size_t i = 1; i < n - 1; ++i) {
        network[i] = Interpolation(i, 0, n - 1, network[0], network[n - 1]); // 0-1
        network[i * n + n - 1] = Interpolation(i, 0, n - 1, network[n - 1], network[n * n - 1]); // 1-2
        network[n * (n - 1) + i] = Interpolation(i, 0, n - 1, network[n * (n - 1)], network[n * n - 1]); // 3-2
        network[i * n] = Interpolation(i, 0, n - 1, network[0], network[n * (n - 1)]); // 0-3
    }
}

void MatrixVectorProduct(const std::vector<double> &matrix, const std::vector<double> &vector,
                         std::vector<double> &result_vector) {
    size_t n = vector.size();
#pragma acc data copyin(matrix[0:n*n], vector[0:n]) copyout(result_vector[0:n])
    {
#pragma acc parallel kernels
        for (size_t i = 0; i < n; ++i) {
            result_vector[i] = 0;
            for (size_t j = 0; j < n; ++j)
                result_vector[i] += matrix[i * n + j] * vector[j];
        }
    }
}

void VectorSubtraction(const std::vector<double> &vector0, const std::vector<double> &vector1,
                       std::vector<double> &result_vector) {
    size_t n = vector0.size();
#pragma acc data copyin(vector0[0:n], vector1[0:n]) copyout(result_vector[0:n])
    {
#pragma acc parallel loop
        for (size_t i = 0; i < n; ++i)
            result_vector[i] = vector0[i] - vector1[i];
    }
}

void ScalarVectorProduct(double scalar, const std::vector<double> &vector, std::vector<double> &result_vector) {
    size_t n = vector.size();
#pragma acc data copyin(vector[0:n]) copyout(result_vector[0:n])
    {
#pragma acc parallel loop
        for (int i = 0; i < n; ++i)
            result_vector[i] = scalar * vector[i];
    }
}


double Norm(const std::vector<double> &vector) {
    size_t n = vector.size();
    double norm{};
#pragma acc data copyin(vector[0:n]) copy(norm)
    {
#pragma acc parallel loop reduction(+:norm)
        for (size_t i = 0; i < n; ++i)
            norm += vector[i] * vector[i];
    }
    return sqrt(norm);
}

void Solve(const std::vector<double> &A, std::vector<double> &x, const std::vector<double> &b,
           double eps, size_t n_max_iterations) {
    size_t n = b.size();
    std::vector<double> buffer_vector(n);
    for (size_t i{}; ; ++i) {
        MatrixVectorProduct(A, x, buffer_vector);
        VectorSubtraction(buffer_vector, b, buffer_vector);
        if (Norm(buffer_vector) < eps * Norm(b))
            break;
        ScalarVectorProduct(-0.000001, buffer_vector, buffer_vector);
        VectorSubtraction(x, buffer_vector, x);
    }
}

int main(int argc, char **argv) {
    double eps = std::stod(argv[1]);
    size_t n = atoi(argv[2]);
    size_t n_max_iterations = atoi(argv[3]);

    std::vector<double> matrix(n * n * n * n);
    std::vector<double> network(n * n, 0);
    std::vector<double> x(n * n, 0);

    InitMatrix(matrix, n);
    PrintMatrix(matrix, n * n);
    InitNetwork(network, n, -10, -20, -30, -20);
    PrintMatrix(network, n);

    Solve(matrix, x, network, eps, n_max_iterations);
    PrintMatrix(x, n);

    return 0;
}
