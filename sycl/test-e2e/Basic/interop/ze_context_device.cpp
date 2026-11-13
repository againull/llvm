// REQUIRES: level_zero, level_zero_dev_kit

// UNSUPPORTED: arch-intel_gpu_mtl_u
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20898

// RUN: %{build} -o %t.out %level_zero_options
// RUN: %{run} %t.out

// This test checks that an interop Level Zero device is properly handled during
// interop context construction.
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/detail/core.hpp>

#include <level_zero/ze_api.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check_ze(ze_result_t Status) {
  if (Status != ZE_RESULT_SUCCESS)
    throw std::runtime_error("Level Zero API failed: " +
                             std::to_string(static_cast<unsigned int>(Status)));
}

int main(int argc, char *argv[]) {
  int level0DriverIndex = 0;
  int level0DeviceIndex = 0;

  check_ze(zeInit(0));

  uint32_t level0NumDrivers = 0;
  check_ze(zeDriverGet(&level0NumDrivers, nullptr));

  if (level0NumDrivers == 0)
    return 1;

  std::vector<ze_driver_handle_t> level0Drivers(level0NumDrivers);
  check_ze(zeDriverGet(&level0NumDrivers, level0Drivers.data()));

  ze_driver_handle_t level0Driver = level0Drivers[level0DriverIndex];
  uint32_t level0NumDevices = 0;
  check_ze(zeDeviceGet(level0Driver, &level0NumDevices, nullptr));

  if (level0NumDevices == 0)
    return 1;

  std::vector<ze_device_handle_t> level0Devices(level0NumDevices);
  check_ze(zeDeviceGet(level0Driver, &level0NumDevices, level0Devices.data()));

  ze_device_handle_t level0Device = level0Devices[level0DeviceIndex];
  ze_context_handle_t level0Context = nullptr;
  ze_context_desc_t level0ContextDesc = {};
  level0ContextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  check_ze(zeContextCreateEx(level0Driver, &level0ContextDesc, 1, &level0Device,
                             &level0Context));

  {
    sycl::device dev;
    sycl::device interopDev =
        sycl::make_device<sycl::backend::ext_oneapi_level_zero>(level0Device);
    sycl::context interopCtx =
        sycl::make_context<sycl::backend::ext_oneapi_level_zero>(
            {level0Context,
             {interopDev},
             sycl::ext::oneapi::level_zero::ownership::keep});

    assert(interopCtx.get_devices().size() == 1);
    assert(interopCtx.get_devices()[0] == interopDev);
    sycl::queue q{interopCtx, interopDev};
  }

  check_ze(zeContextDestroy(level0Context));
  return 0;
}
