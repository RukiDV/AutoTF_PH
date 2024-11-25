#pragma once
#include <cstdint>
#include <limits>
#include <random>

class RandomGenerator
{
public:
  RandomGenerator(uint32_t seed = 42);
  std::mt19937& get_generator();
  float random_float(float lower_bound = 0.0f, float upper_bound = 1.0f);
  int32_t random_int32(int32_t lower_bound = std::numeric_limits<int32_t>::min(), int32_t upper_bound = std::numeric_limits<int32_t>::max());

private:
  std::uniform_real_distribution<float> distribution;
  std::mt19937 generator;
};

namespace rng
{
// none of these functions is thread safe
// if you require thread safe random numbers create a RandomGenerator for every thread (with a different seed)

float random_float(float lower_bound = 0.0f, float upper_bound = 1.0f);

// generate random int in [lower_bound, upper_bound - 1]
int32_t random_int32(int32_t lower_bound = std::numeric_limits<int32_t>::min(), int32_t upper_bound = std::numeric_limits<int32_t>::max());

// this function should not be used most of the time
// instead create your own instance of the RandomGenerator
RandomGenerator& get_instance();
} // namespace rng
