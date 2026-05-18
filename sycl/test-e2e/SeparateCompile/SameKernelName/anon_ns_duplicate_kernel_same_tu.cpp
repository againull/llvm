// Test: Two kernels with the same name in one translation unit, both kernel
// name types enclosed in anonymous namespace. Since they are in the same TU's
// anonymous namespace, this is the SAME type used for two different kernels.
// The SYCL device compiler must generate an error.
//
// XFAIL: *
// XFAIL-TRACKER: TBD — compiler should reject duplicate kernel names in same TU
//
// RUN: %{build} -fsyntax-only 2>&1 | FileCheck %s

// CHECK: {{error|warning}}:{{.*}}kernel{{.*}}name{{.*}}conflict

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

namespace {
struct KN;
}

int main() {
  sycl::queue q;
  constexpr int N = 16;
  int *out1 = sycl::malloc_device<int>(N, q);
  int *out2 = sycl::malloc_device<int>(N, q);

  // First kernel using KN
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KN>(
        sycl::range<1>(N), [=](sycl::id<1> i) { out1[i] = static_cast<int>(i[0]); });
  });

  // Second kernel using the same KN type — this should be a compile error
  q.submit([&](sycl::handler &h) {
    h.parallel_for<KN>(sycl::range<1>(N), [=](sycl::id<1> i) {
      out2[i] = static_cast<int>(i[0]) * 2;
    });
  });

  q.wait();
  sycl::free(out1, q);
  sycl::free(out2, q);
  return 0;
}
