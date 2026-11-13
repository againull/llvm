// REQUIRES: aspect-fp16
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>
#include <sycl/detail/vector_convert.hpp>

#include <array>
#include <cmath>
#include <limits>

using namespace sycl;

constexpr int SZ_max = 16;

bool check(float a, float b) {
  return sycl::fabs(2 * (a - b) / (a + b)) <
             std::numeric_limits<half>::epsilon() ||
         a < std::numeric_limits<half>::min();
}

template <int N> bool check(vec<float, N> a, vec<float, N> b) {
  for (int i = 0; i < N; i++) {
    if (!check(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

template <int N> vec<float, N> load_packet(const float16 &Input, size_t Index) {
  constexpr size_t Stride = N == 3 ? 4 : N;
  vec<float, N> Result;
  for (size_t Lane = 0; Lane < N; ++Lane)
    Result[Lane] = Input[Index * Stride + Lane];
  return Result;
}

#define TEST_BUILTIN_1_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    if (i < SZ_max / (SZ == 3 ? 4 : SZ)) {                                     \
      auto a = load_packet<SZ>(A[0], i);                                       \
      if (!check(NAME(a), NAME(a.convert<half>()).convert<float>())) {         \
        err[i] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

// vectors of size 3 need separate test, as they actually have the size of 4
// elements
#define TEST_BUILTIN_1_VEC3_IMPL(NAME) TEST_BUILTIN_1_VEC_IMPL(NAME, 3)

#define TEST_BUILTIN_1_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    float a = A[0][i];                                                         \
    if (!check(NAME(a), (float)NAME((half)a))) {                               \
      err[i] = 1;                                                              \
    }                                                                          \
  }

#define TEST_BUILTIN_1(NAME)                                                   \
  TEST_BUILTIN_1_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_1_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_1_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_2_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    if (i < SZ_max / (SZ == 3 ? 4 : SZ)) {                                     \
      auto a = load_packet<SZ>(A[0], i);                                       \
      auto b = load_packet<SZ>(B[0], i);                                       \
      if (!check(                                                              \
              NAME(a, b),                                                      \
              NAME(a.convert<half>(), b.convert<half>()).convert<float>())) {  \
        err[i] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_2_VEC3_IMPL(NAME) TEST_BUILTIN_2_VEC_IMPL(NAME, 3)

#define TEST_BUILTIN_2_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    float a = A[0][i];                                                         \
    float b = B[0][i];                                                         \
    if (!check(NAME(a, b), (float)NAME((half)a, (half)b))) {                   \
      err[i] = 1;                                                              \
    }                                                                          \
  }

#define TEST_BUILTIN_2(NAME)                                                   \
  TEST_BUILTIN_2_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_2_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_2_VEC_IMPL(NAME, 16)

#define TEST_BUILTIN_3_VEC_IMPL(NAME, SZ)                                      \
  {                                                                            \
    if (i < SZ_max / (SZ == 3 ? 4 : SZ)) {                                     \
      auto a = load_packet<SZ>(A[0], i);                                       \
      auto b = load_packet<SZ>(B[0], i);                                       \
      auto c = load_packet<SZ>(C[0], i);                                       \
      if (!check(NAME(a, b, c),                                                \
                 NAME(a.convert<half>(), b.convert<half>(), c.convert<half>()) \
                     .convert<float>())) {                                     \
        err[i] = 1;                                                            \
      }                                                                        \
    }                                                                          \
  }

#define TEST_BUILTIN_3_VEC3_IMPL(NAME) TEST_BUILTIN_3_VEC_IMPL(NAME, 3)

#define TEST_BUILTIN_3_SCAL_IMPL(NAME)                                         \
  {                                                                            \
    float a = A[0][i];                                                         \
    float b = B[0][i];                                                         \
    float c = C[0][i];                                                         \
    if (!check(NAME(a, b, c), (float)NAME((half)a, (half)b, (half)c))) {       \
      err[i] = 1;                                                              \
    }                                                                          \
  }

#define TEST_BUILTIN_3(NAME)                                                   \
  TEST_BUILTIN_3_SCAL_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 2)                                             \
  TEST_BUILTIN_3_VEC3_IMPL(NAME)                                               \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 4)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 8)                                             \
  TEST_BUILTIN_3_VEC_IMPL(NAME, 16)

int main() {
  queue q;

  float16 a, b, c, d;
  for (int i = 0; i < SZ_max; i++) {
    a[i] = i / (float)SZ_max;
    b[i] = (SZ_max - i) / (float)SZ_max;
    c[i] = (float)(3 * i);
  }
  std::array<int, SZ_max> errors{};
  {
    buffer<float16> a_buf(&a, 1);
    buffer<float16> b_buf(&b, 1);
    buffer<float16> c_buf(&c, 1);
    buffer<int> err_buf(errors.data(), range<1>(SZ_max));
    q.submit([&](handler &cgh) {
      auto A = a_buf.get_access<access_mode::read>(cgh);
      auto B = b_buf.get_access<access_mode::read>(cgh);
      auto C = c_buf.get_access<access_mode::read>(cgh);
      auto err = err_buf.get_access<access_mode::write>(cgh);
      cgh.parallel_for(SZ_max, [=](item<1> index) {
        size_t i = index.get_id(0);
        err[i] = 0;
        TEST_BUILTIN_1(sycl::fabs);
        TEST_BUILTIN_2(sycl::fmin);
        TEST_BUILTIN_2(sycl::fmax);
        TEST_BUILTIN_3(sycl::fma);
      });
    });
  }
  for (int error : errors)
    assert(error == 0);

  return 0;
}
