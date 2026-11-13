// RUN: %{build} -o %t.out -DSYCL_DISABLE_IMAGE_ASPECT_WARNING
// RUN: %{run} %t.out

// Check that platform aspects agree with the intersection of device aspects.
#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>

#include <algorithm>
#include <cassert>

int main() {
  const auto Platform = sycl::device{}.get_platform();
  const auto Devices = Platform.get_devices();
  const sycl::aspect Aspects[] = {sycl::aspect::cpu,
                                  sycl::aspect::gpu,
                                  sycl::aspect::accelerator,
                                  sycl::aspect::custom,
                                  sycl::aspect::fp16,
                                  sycl::aspect::fp64,
                                  sycl::aspect::atomic64,
                                  sycl::aspect::image,
                                  sycl::aspect::online_compiler,
                                  sycl::aspect::online_linker,
                                  sycl::aspect::queue_profiling,
                                  sycl::aspect::usm_device_allocations,
                                  sycl::aspect::usm_host_allocations,
                                  sycl::aspect::usm_atomic_host_allocations,
                                  sycl::aspect::usm_shared_allocations,
                                  sycl::aspect::usm_atomic_shared_allocations,
                                  sycl::aspect::usm_system_allocations,
                                  sycl::aspect::ext_oneapi_cuda_cluster_group,
                                  sycl::aspect::ext_oneapi_atomic16,
                                  sycl::aspect::ext_oneapi_virtual_functions,
                                  sycl::aspect::ext_intel_spill_memory_size};
  assert(!Devices.empty());
  for (auto Aspect : Aspects) {
    const bool Expected =
        std::all_of(Devices.begin(), Devices.end(),
                    [=](const sycl::device &Dev) { return Dev.has(Aspect); });
    assert(Platform.has(Aspect) == Expected);
  }
}
