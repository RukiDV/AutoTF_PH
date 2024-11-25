#pragma once
#include <iostream>
#include <stdexcept>

#define VE_C_RED "\033[91m"
#define VE_C_GREEN "\033[92m"
#define VE_C_YELLOW "\033[93m"
#define VE_C_BLUE "\033[94m"
#define VE_C_PINK "\033[95m"
#define VE_C_LBLUE "\033[96m"
#define VE_C_WHITE "\033[0m"

#define VE_CHECKING

#define VE_THROW(...)                    \
{                                        \
  std::cerr << __VA_ARGS__ << std::endl; \
  std::string s(__FILE__);               \
  s.append(": ");                        \
  s.append(std::to_string(__LINE__));    \
  throw std::runtime_error(s);       \
}

#if defined(VE_CHECKING)
#define VE_ASSERT(X, ...) if (!(X)) VE_THROW(__VA_ARGS__);
#define VE_CHECK(X, M) vk::detail::resultCheck(X, M)
#else
#define VE_ASSERT(X, ...) X
#define VE_CHECK(X, M) void(X)
#endif
