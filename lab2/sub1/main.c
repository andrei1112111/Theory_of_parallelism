#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


void matrix_product(const double *a, const double *b, double *c, int m, int n) {
    // calc
    for (int i = 0; i < m; ++i) {
        c[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void matrix_product_omp(const double *a, const double *b, double *c, int m, int n, int threads) {
    #pragma omp parallel num_threads(threads) 
    {
        int nThreads = omp_get_num_threads();
        int threadId = omp_get_thread_num();

        // find range for current thread
        int items_per_thread = m / nThreads;

        int lower_b = threadId * items_per_thread;

        int upper_b;
        if (threadId == nThreads - 1) {
            upper_b = m - 1;
        } else {
            upper_b = lower_b + items_per_thread - 1;
        }

        // calc
        for (int i = lower_b; i <= upper_b; ++i) {
            c[i] = 0.0;
            for (int j = 0; j < n; ++j) {
                c[i] += a[i * n + j] * b[j];
            }
        }
    }
}

void calculate_serial(int m, int n, int iterations) {
    double *a, *b, *c;
    a = (double *) malloc(sizeof(*a) * m * n);  // a[m, n]
    b = (double *) malloc(sizeof(*b) * n);  // b[n]
    c = (double *) malloc(sizeof(*c) * m);  // c[m]

    // fill a and b with some numbers
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = i + j;
        }
    }
    for (int j = 0; j < n; ++j) {
        b[j] = j;
    }

    double start_time = omp_get_wtime();

    for (int i = 0; i < iterations; ++i) {
        matrix_product(a, b, c, m, n);  // calculate
    }

    double elapsed_time = omp_get_wtime() - start_time;

    printf("(serial) On average, %d iterations took %.3f sec.\n", iterations, elapsed_time / iterations);

    free(a);
    free(b);
    free(c);
}

void calculate_parallel(int m, int n, int threads, int iterations) {
    double *a, *b, *c;
    a = (double *) malloc(sizeof(*a) * m * n);  // a[m, n]
    b = (double *) malloc(sizeof(*b) * n);  // b[n]
    c = (double *) malloc(sizeof(*c) * m);  // c[m]

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = i + j;
        }
    }
    for (int j = 0; j < n; ++j) {
        b[j] = j;
    }

    double start_time = omp_get_wtime();

    for (int i = 0; i < iterations; ++i) {
        matrix_product_omp(a, b, c, m, n, threads);
    }

    double elapsed_time = omp_get_wtime() - start_time;

    printf("(parallel) On average, %d iterations using %d threads took %.3f sec.\n", iterations, threads, elapsed_time / iterations);

    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "4 arguments are needed.\n");
        return -1;
    }
       
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int iterations = atoi(argv[4]);

    printf("Matrix-vector product of c[%d] = a[%d, %d] * b[%d]\n", m, m, n, n);
    printf("Memory using: %lu MiB.\n", ((m * n + m + n) * sizeof(double)) / (1024 * 1024));

    // calculate_serial(m, n, iterations);
    calculate_parallel(m, n, threads, iterations);

    return 0;
}
