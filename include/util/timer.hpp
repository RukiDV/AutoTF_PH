#pragma once

#include <chrono>

constexpr const char* CLR_RED    = "\033[31m";
constexpr const char* CLR_GREEN  = "\033[32m";
constexpr const char* CLR_YELLOW = "\033[33m";
constexpr const char* CLR_BLUE   = "\033[34m";
constexpr const char* CLR_RESET  = "\033[0m";

template<class Precision = float>
class Timer
{
public:
    Timer() : t(std::chrono::high_resolution_clock::now())
    {}

    // default timer unit is seconds
    template<class Period = std::ratio<1, 1>>
    inline Precision restart()
    {
        Precision elapsed_time = elapsed<Period>();
        t = std::chrono::high_resolution_clock::now();
        return elapsed_time;
    }

    template<class Period = std::ratio<1, 1>>
    inline Precision elapsed() const
    {
        return std::chrono::duration<Precision, Period>(std::chrono::high_resolution_clock::now() - t).count();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t;
};
