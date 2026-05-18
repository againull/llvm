// Header that is included with different macro definitions in different TUs,
// resulting in the same kernel name but different kernel parameters.
// The device linker should emit an error in this case.

#pragma once

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

struct SharedKernelDiffParams;

#ifdef USE_FLOAT
inline void run_kernel_diff_params(sycl::queue &q, float *out, int N) {
  q.submit([&](sycl::handler &h) {
     h.parallel_for<SharedKernelDiffParams>(
         sycl::range<1>(N),
         [=](sycl::id<1> i) { out[i] = static_cast<float>(i[0]) * 2.0f; });
   }).wait();
}
#else
inline void run_kernel_diff_params(sycl::queue &q, int *out, int N) {
  q.submit([&](sycl::handler &h) {
     h.parallel_for<SharedKernelDiffParams>(
         sycl::range<1>(N),
         [=](sycl::id<1> i) { out[i] = static_cast<int>(i[0]) * 2; });
   }).wait();
}
#endif
