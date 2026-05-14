// Test that invoking a kernel via kernel_bundle from within each DSO (and
// main) runs that DSO's own version of the kernel. Three compilation units
// define the same KernelFunctor name with identical layout but different
// bodies (+10, +20, +30). Each unit obtains a kernel_bundle and launches
// the kernel through it — the test verifies per-DSO correctness.
//
// REQUIRES: linux
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB_A -fPIC -shared -o %t.dir/lib_a.so
// RUN: %{build} -DBUILD_LIB_B -fPIC -shared -o %t.dir/lib_b.so
// RUN: %{build} -DBUILD_MAIN -DLIB_DIR=%t.dir -ldl -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

class KernelFunctor;

#ifdef BUILD_LIB_A
// KernelFunctor in lib_a: out[i] = in[i] + 10
extern "C" __attribute__((visibility("default"))) int
run_via_bundle_a(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q->get_context(), {q->get_device()},
      {sycl::get_kernel_id<KernelFunctor>()});

  q->submit([&](sycl::handler &cgh) {
     cgh.use_kernel_bundle(bundle);
     cgh.parallel_for<KernelFunctor>(sycl::range<1>(N), [=](sycl::id<1> i) {
       out[i] = in[i] + 10.0f;
     });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 11.0f) {
      std::fprintf(stderr, "run_via_bundle_a: result[%d] = %f, expected 11.0\n",
                   i, result[i]);
      return -1;
    }
  }
  return 0;
}
#endif

#ifdef BUILD_LIB_B
// KernelFunctor in lib_b: out[i] = in[i] + 20
extern "C" __attribute__((visibility("default"))) int
run_via_bundle_b(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q->get_context(), {q->get_device()},
      {sycl::get_kernel_id<KernelFunctor>()});

  q->submit([&](sycl::handler &cgh) {
     cgh.use_kernel_bundle(bundle);
     cgh.parallel_for<KernelFunctor>(sycl::range<1>(N), [=](sycl::id<1> i) {
       out[i] = in[i] + 20.0f;
     });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 21.0f) {
      std::fprintf(stderr, "run_via_bundle_b: result[%d] = %f, expected 21.0\n",
                   i, result[i]);
      return -1;
    }
  }
  return 0;
}
#endif

#ifdef BUILD_MAIN
#include <dlfcn.h>

#define STRINGIFY_HELPER(A) #A
#define STRINGIFY(A) STRINGIFY_HELPER(A)

using fn_t = int (*)(sycl::queue *);

// KernelFunctor in main: out[i] = in[i] + 30
int run_via_bundle_main(sycl::queue *q) {
  constexpr int N = 64;
  float *out = sycl::malloc_device<float>(N, *q);
  float *in = sycl::malloc_device<float>(N, *q);
  q->fill(in, 1.0f, N).wait();
  q->memset(out, 0, N * sizeof(float)).wait();

  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      q->get_context(), {q->get_device()},
      {sycl::get_kernel_id<KernelFunctor>()});

  q->submit([&](sycl::handler &cgh) {
     cgh.use_kernel_bundle(bundle);
     cgh.parallel_for<KernelFunctor>(sycl::range<1>(N), [=](sycl::id<1> i) {
       out[i] = in[i] + 30.0f;
     });
   }).wait();

  std::vector<float> result(N);
  q->memcpy(result.data(), out, N * sizeof(float)).wait();
  sycl::free(out, *q);
  sycl::free(in, *q);

  for (int i = 0; i < N; ++i) {
    if (result[i] != 31.0f) {
      std::fprintf(stderr,
                   "run_via_bundle_main: result[%d] = %f, expected 31.0\n", i,
                   result[i]);
      return -1;
    }
  }
  return 0;
}

int main() {
  sycl::queue q;

  void *h_a =
      dlopen(STRINGIFY(LIB_DIR) "/lib_a.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h_a) {
    std::fprintf(stderr, "Failed to load lib_a.so: %s\n", dlerror());
    return 1;
  }
  void *h_b =
      dlopen(STRINGIFY(LIB_DIR) "/lib_b.so", RTLD_NOW | RTLD_GLOBAL);
  if (!h_b) {
    std::fprintf(stderr, "Failed to load lib_b.so: %s\n", dlerror());
    return 1;
  }

  auto *fn_a = (fn_t)dlsym(h_a, "run_via_bundle_a");
  auto *fn_b = (fn_t)dlsym(h_b, "run_via_bundle_b");
  if (!fn_a || !fn_b) {
    std::fprintf(stderr, "Failed to find symbols\n");
    return 1;
  }

  int rc_a = fn_a(&q);
  int rc_b = fn_b(&q);
  int rc_main = run_via_bundle_main(&q);

  std::printf("lib_a via bundle (expect 11.0): %s\n",
              rc_a == 0 ? "PASS" : "FAIL");
  std::printf("lib_b via bundle (expect 21.0): %s\n",
              rc_b == 0 ? "PASS" : "FAIL");
  std::printf("main via bundle (expect 31.0): %s\n",
              rc_main == 0 ? "PASS" : "FAIL");

  assert(rc_a == 0 && "lib_a kernel_bundle invocation ran wrong kernel");
  assert(rc_b == 0 && "lib_b kernel_bundle invocation ran wrong kernel");
  assert(rc_main == 0 && "main kernel_bundle invocation ran wrong kernel");

  dlclose(h_a);
  dlclose(h_b);

  return 0;
}
#endif
