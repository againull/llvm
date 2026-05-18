// Shared header defining an inline function that launches a kernel.
// Multiple TUs include this header, resulting in multiple definitions
// of the same kernel (same name, same parameters, same body).
// The device linker should merge them (comdat-like behavior).

#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

struct SharedKernel;

inline void run_shared_kernel(sycl::queue &q, int *out, int N) {
  q.submit([&](sycl::handler &h) {
     h.parallel_for<SharedKernel>(
         sycl::range<1>(N),
         [=](sycl::id<1> i) { out[i] = static_cast<int>(i[0]) * 3; });
   }).wait();
}
