// Test: Same kernel name (in anonymous namespace) across two translation units.
// Each TU defines a kernel named KN in its own anonymous namespace. The runtime
// must dispatch to the correct kernel based on the originating TU.
//
// RUN: %{build} -DBUILD_K1 -c -o %t-k1.o
// RUN: %{build} -DBUILD_K2 -c -o %t-k2.o
// RUN: %{build} -DBUILD_MAIN %t-k1.o %t-k2.o -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>

namespace {
struct KN;
}

#ifdef BUILD_K1
void k1(sycl::queue &q) {
  constexpr int N = 16;
  int *out = sycl::malloc_device<int>(N, q);
  q.submit([&](sycl::handler &h) {
     h.parallel_for<KN>(sycl::range<1>(N), [=](sycl::id<1> i) {
       out[i] = static_cast<int>(i[0]) + 1;
     });
   }).wait();

  std::vector<int> result(N);
  q.memcpy(result.data(), out, N * sizeof(int)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == i + 1 && "k1 kernel produced wrong result");
  }
  std::printf("k1: PASS\n");
}
#endif

#ifdef BUILD_K2
void k2(sycl::queue &q) {
  constexpr int N = 16;
  int *out = sycl::malloc_device<int>(N, q);
  q.submit([&](sycl::handler &h) {
     h.parallel_for<KN>(sycl::range<1>(N), [=](sycl::id<1> i) {
       out[i] = (static_cast<int>(i[0]) + 1) * 100;
     });
   }).wait();

  std::vector<int> result(N);
  q.memcpy(result.data(), out, N * sizeof(int)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == (i + 1) * 100 && "k2 kernel produced wrong result");
  }
  std::printf("k2: PASS\n");
}
#endif

#ifdef BUILD_MAIN
void k1(sycl::queue &q);
void k2(sycl::queue &q);

int main() {
  sycl::queue q;
  k1(q);
  k2(q);
  return 0;
}
#endif
