#include <iostream>
#include <boost/program_options.hpp>
#include <thread>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>


std::chrono::duration<double> Multiplication(int N, int threadAmount) {
    std::vector<std::thread> threads(threadAmount);
    std::vector<double> matrix(N * N);
    std::vector<double> vector(N);

    int itemsPerThread = N / threadAmount;

    for (int i = 0; i < threads.size(); ++i) {
        int lb = i * itemsPerThread;
        int ub = (i == threads.size() - 1) ? (N - 1) : (lb + itemsPerThread - 1);

        threads[i] = std::thread([](std::vector<double> &matrix, std::vector<double> &vector, int lb, int ub) {
            for (int i = lb; i <= ub; ++i) {
                for (int j = 0; j < vector.size(); ++j)
                    matrix[i * vector.size() + j] = i + j;
                vector[i] = i;
            }
        }, std::ref(matrix), std::ref(vector), lb, ub);
    }
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

    std::vector<double> resultVector(N, 0);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < threads.size(); ++i) {
        int lb = i * itemsPerThread;
        int ub = (i == threads.size() - 1) ? (N - 1) : (lb + itemsPerThread - 1);

        threads[i] = std::thread(
                [&resultVector](const std::vector<double> &matrix, const std::vector<double> &vector, int lb, int ub) {
                    for (int i = lb; i <= ub; ++i)
                        for (int j = 0; j < vector.size(); ++j) {
                            resultVector[i] += matrix[i * vector.size() + j] * vector[j];
                        }
                }, std::cref(matrix), std::cref(vector), lb, ub);
    }
    
    std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
    const auto end = std::chrono::steady_clock::now();

    return (end - start);
}

int main(int argc, char **argv) {
    try {
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "Show help message")
            ("size,N", boost::program_options::value<int>()->required(), "Matrix size")
            ("threads,T", boost::program_options::value<int>()->required(), "Number of threads");
        
        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        
        boost::program_options::notify(vm);
        
        int N = vm["size"].as<int>();
        int threadAmount = vm["threads"].as<int>();
        
        if (N <= 0 || threadAmount <= 0) {
            std::cerr << "Error: Matrix size and thread count must be positive integers." << std::endl;
            return 1;
        }
        
        auto duration = Multiplication(N, threadAmount);
        
        std::cout << "Execution time: " << std::fixed << duration.count() << " seconds" << std::endl;
    } catch (const boost::program_options::error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
