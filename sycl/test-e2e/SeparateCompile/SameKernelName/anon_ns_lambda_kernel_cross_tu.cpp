// Test: Same unnamed lambda kernel name across two translation units.
// Each TU defines a static function with the same name that invokes an unnamed
// lambda kernel. The lambda closure types get the same nested name in each TU.
// The runtime must dispatch to the correct kernel based on the originating TU.
//
// RUN: %{build} -DBUILD_K1 -c -o %t-k1.o
// RUN: %{build} -DBUILD_K2 -c -o %t-k2.o
// RUN: %{build} -DBUILD_MAIN %t-k1.o %t-k2.o -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>

#ifdef BUILD_K1
static void run_kernel(sycl::queue &q, int *out, int N) {
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range<1>(N),
                    [=](sycl::id<1> i) { out[i] = static_cast<int>(i[0]) + 1; });
   }).wait();
}

void k1(sycl::queue &q) {
  constexpr int N = 16;
  int *out = sycl::malloc_device<int>(N, q);
  run_kernel(q, out, N);

  std::vector<int> result(N);
  q.memcpy(result.data(), out, N * sizeof(int)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == i + 1 && "k1 lambda kernel produced wrong result");
  }
  std::printf("k1: PASS\n");
}
#endif

#ifdef BUILD_K2
static void run_kernel(sycl::queue &q, int *out, int N) {
  q.submit([&](sycl::handler &h) {
     h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
       out[i] = (static_cast<int>(i[0]) + 1) * 100;
     });
   }).wait();
}

void k2(sycl::queue &q) {
  constexpr int N = 16;
  int *out = sycl::malloc_device<int>(N, q);
  run_kernel(q, out, N);

  std::vector<int> result(N);
  q.memcpy(result.data(), out, N * sizeof(int)).wait();
  sycl::free(out, q);

  for (int i = 0; i < N; ++i) {
    assert(result[i] == (i + 1) * 100 && "k2 lambda kernel produced wrong result");
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
