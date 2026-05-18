// Test: Inline function in header defining a kernel, included from multiple
// translation units. This is analogous to normal C++ inline/ODR rules: multiple
// definitions are allowed if they have the same signature and body. The device
// linker should merge them like comdat.
//
// RUN: %{build} -DBUILD_TU1 -c -o %t-tu1.o
// RUN: %{build} -DBUILD_TU2 -c -o %t-tu2.o
// RUN: %{build} -DBUILD_MAIN %t-tu1.o %t-tu2.o -o %t.out
// RUN: %{run} %t.out

#include "inline_header_same_kernel.hpp"
#include <cassert>
#include <cstdio>
#include <vector>

#ifdef BUILD_TU1
void test_from_tu1(sycl::queue &q) {
  constexpr int N = 16;
  int *out = sycl::malloc_device<int>(N, q);
  run_shared_kernel(q, out, N);

  std::vector<int> result(N);
  q.memcpy(result.data(), out, N * sizeof(int)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == i * 3 && "test_from_tu1 failed");
  }
  std::printf("tu1: PASS\n");
}
#endif

#ifdef BUILD_TU2
void test_from_tu2(sycl::queue &q) {
  constexpr int N = 16;
  int *out = sycl::malloc_device<int>(N, q);
  run_shared_kernel(q, out, N);

  std::vector<int> result(N);
  q.memcpy(result.data(), out, N * sizeof(int)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == i * 3 && "test_from_tu2 failed");
  }
  std::printf("tu2: PASS\n");
}
#endif

#ifdef BUILD_MAIN
void test_from_tu1(sycl::queue &q);
void test_from_tu2(sycl::queue &q);

int main() {
  sycl::queue q;
  test_from_tu1(q);
  test_from_tu2(q);
  return 0;
}
#endif
