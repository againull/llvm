// Test: Same kernel name from header included in multiple TUs, but with
// different kernel parameters (due to different macro definitions).
// The device linker must emit an error because the kernel signatures differ.
//
// XFAIL: *
// XFAIL-TRACKER: TBD — device linker should reject mismatched kernel signatures
//
// RUN: %{build} -DBUILD_TU1 -DUSE_FLOAT -c -o %t-tu1.o
// RUN: %{build} -DBUILD_TU2 -c -o %t-tu2.o
// RUN: not %{build} -DBUILD_MAIN %t-tu1.o %t-tu2.o -o %t.out 2>&1 | FileCheck %s

// CHECK: {{error}}

#include "inline_header_different_params.hpp"
#include <cassert>
#include <cstdio>
#include <vector>

#ifdef BUILD_TU1
void test_float(sycl::queue &q) {
  constexpr int N = 16;
  float *out = sycl::malloc_device<float>(N, q);
  run_kernel_diff_params(q, out, N);

  std::vector<float> result(N);
  q.memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == static_cast<float>(i) * 2.0f && "test_float failed");
  }
  std::printf("float: PASS\n");
}
#endif

#ifdef BUILD_TU2
void test_int(sycl::queue &q) {
  constexpr int N = 16;
  int *out = sycl::malloc_device<int>(N, q);
  run_kernel_diff_params(q, out, N);

  std::vector<int> result(N);
  q.memcpy(result.data(), out, N * sizeof(int)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == i * 2 && "test_int failed");
  }
  std::printf("int: PASS\n");
}
#endif

#ifdef BUILD_MAIN
void test_float(sycl::queue &q);
void test_int(sycl::queue &q);

int main() {
  sycl::queue q;
  test_float(q);
  test_int(q);
  return 0;
}
#endif
