#include <iostream>
#include <queue>
#include <functional>
#include <thread>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

enum Task {
    Sin,
    Sqrt,
    Pow
};

template<typename T>
T fun_sin(T x, T y) {
    return std::sin(x);
}

template<typename T>
T fun_sqrt(T x, T y) {
    return std::sqrt(x);
}

template<typename T>
T fun_pow(T x, T y) {
    return std::pow(x, y);
}

namespace Test {
    template<typename T>
    bool IsCorrect(std::ifstream &inputStream) {
        std::unordered_map<std::string, Task> map;
        map["Sin"] = Sin;
        map["Sqrt"] = Sqrt;
        map["Pow"] = Pow;
        size_t id{};
        std::string taskString{};
        Task taskType{};
        T res{}, x{}, y{}, a{};
        bool okFlag{true};
        while (inputStream && okFlag) {
            inputStream >> id >> taskString;

            auto it = map.find(taskString);
            if (it == map.end()) {
                okFlag = false;  // Некорректный ввод
                continue;
            }
            taskType = it->second;

            if (taskType == Pow)
                inputStream >> res >> x >> y;
            else
                inputStream >> res >> x;
            switch (taskType) {
                case Sin: {
                    T awaited = fun_sin(x, static_cast<T>(0));
                    if (std::abs(res - awaited) > 0.1)
                        okFlag = false;
                    break;
                }
                case Sqrt: {
                    T awaited = fun_sqrt(x, static_cast<T>(0));
                    if (std::abs(res - awaited) > 0.1)
                        okFlag = false;
                    break;
                }
                case Pow: {
                    T awaited = fun_pow(x, y);
                    if (std::abs(res - awaited) > 0.1)
                        okFlag = false;
                    break;
                }
            }
        }
        return okFlag;
    }
}

template<typename T>
class Server {
private:
    std::mutex queueMutex;
    std::queue<std::tuple<Task, T, T>> taskQueue;
    std::vector<std::tuple<std::string, T, T, T>> results;
    std::jthread serverThread;
    std::condition_variable cv;

    std::unordered_map<Task, std::function<T(T, T)>> funcMap{
        {Sin, fun_sin<T>},
        {Sqrt, fun_sqrt<T>},
        {Pow, fun_pow<T>}
    };

    std::unordered_map<Task, std::string> taskNames{
        {Sin, "Sin"},
        {Sqrt, "Sqrt"},
        {Pow, "Pow"}
    };

    void Work(const std::stop_token &stopToken) {
        while (!stopToken.stop_requested()) {
            std::unique_lock lock(queueMutex);
            cv.wait(lock, [this] { return !taskQueue.empty(); });

            auto [taskType, x, y] = taskQueue.front();
            taskQueue.pop();
            lock.unlock();

            T result = funcMap[taskType](x, y);
            results.emplace_back(taskNames[taskType], result, x, y);
        }
    }

public:
    Server() : serverThread{} {}

    void Start() {
        serverThread = std::jthread([this](std::stop_token stopToken) { Work(stopToken); });
    }

    void Stop() { serverThread.request_stop(); }

    void Join() { serverThread.join(); }

    void AddTask(Task taskType, T x, T y) {
        std::lock_guard lock(queueMutex);
        taskQueue.emplace(taskType, x, y);
        cv.notify_one();
    }

    std::vector<std::tuple<std::string, T, T, T>> GetResult() {
        return std::move(results);
    }
};

template<typename T>
class Client {
public:
    Client(Server<T> &server, Task taskType) : server(server), taskType(taskType) {}

    void Start(int N) {
        thread = std::thread([N, this] {
            for (int i = 0; i < N; ++i)
                server.AddTask(taskType, rand() % 100, rand() % 4);
        });
    }

    void Join() { thread.join(); }

private:
    Server<T> &server;
    Task taskType;
    std::thread thread;
};

template<typename T>
std::vector<std::tuple<std::string, T, T, T>> Process() {
    auto server = std::make_shared<Server<T>>();
    auto client_sin = std::make_unique<Client<T>>(*server, Sin);
    auto client_sqrt = std::make_unique<Client<T>>(*server, Sqrt);
    auto client_pow = std::make_unique<Client<T>>(*server, Pow);

    server->Start();

    client_sin->Start(10000);
    client_sqrt->Start(10000);
    client_pow->Start(10000);

    client_sin->Join();
    client_sqrt->Join();
    client_pow->Join();

    server->Stop();
    server->Join();

    return std::move(server->GetResult());
}

int main() {
    std::srand(std::time(nullptr));
    auto res = Process<double>();
    std::ofstream outputFile;
    outputFile.open("output.txt");
    outputFile << std::fixed << std::setprecision(5);
    for (size_t i{}; i < res.size(); ++i) {
        if (get<0>(res[i]) == "Pow")
            outputFile << i << " "
                       << get<0>(res[i]) << " "
                       << get<1>(res[i]) << " "
                       << get<2>(res[i]) << " "
                       << get<3>(res[i]) << std::endl;
        else
            outputFile << i << " "
                       << get<0>(res[i]) << " "
                       << get<1>(res[i]) << " "
                       << get<2>(res[i]) << std::endl;
    }
    outputFile.close();
    std::ifstream testInput("output.txt");
    if (Test::IsCorrect<double>(testInput))
        std::cout << "Everything is correct!" << std::endl;
    else
        std::cout << "Something went wrong!" << std::endl;
    return 0;
}
