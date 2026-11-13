//==----- DeviceGetInfoAspects.cpp --- info::device::aspects unit test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <algorithm>

using namespace sycl;

static bool containsAspect(const std::vector<sycl::aspect> &DeviceAspects,
                           const aspect Aspect) {
  return std::find(DeviceAspects.begin(), DeviceAspects.end(), Aspect) !=
         DeviceAspects.end();
}

TEST(DeviceGetInfo, SupportedDeviceAspects) {
  sycl::unittest::UrMock<> Mock;

  sycl::platform Plt = sycl::platform();
  sycl::device Dev = Plt.get_devices()[0];

  std::vector<sycl::aspect> DeviceAspects =
      Dev.get_info<info::device::aspects>();

  // Tests to examine aspects of default mock device, as defined in
  // helpers/UrMockAdapter.hpp so these tests all need to be kept in sync with
  // changes to that file.
  EXPECT_TRUE(containsAspect(DeviceAspects, aspect::gpu));
  EXPECT_TRUE(containsAspect(DeviceAspects, aspect::fp16));
  EXPECT_TRUE(containsAspect(DeviceAspects, aspect::fp64));

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
  EXPECT_FALSE(containsAspect(DeviceAspects, aspect::host));
#endif // __INTEL_PREVIEW_BREAKING_CHANGES
  EXPECT_FALSE(containsAspect(DeviceAspects, aspect::cpu));
}

TEST(DeviceGetInfo, PlatformAspectsRequireAllDevices) {
  sycl::unittest::UrMock<> Mock;
  mock::getCallbacks().set_replace_callback("urDeviceGet", [](void *Params) {
    auto &P = *static_cast<ur_device_get_params_t *>(Params);
    if (*P.ppNumDevices)
      **P.ppNumDevices = 2;
    if (*P.pphDevices) {
      for (uint32_t I = 0; I < std::min(*P.pNumEntries, 2u); ++I)
        (*P.pphDevices)[I] = reinterpret_cast<ur_device_handle_t>(I + 1);
    }
    return UR_RESULT_SUCCESS;
  });
  mock::getCallbacks().set_after_callback("urDeviceGetInfo", [](void *Params) {
    auto &P = *static_cast<ur_device_get_info_params_t *>(Params);
    if (*P.ppPropValue) {
      if (*P.ppropName == UR_DEVICE_INFO_TYPE)
        *static_cast<ur_device_type_t *>(*P.ppPropValue) =
            *P.phDevice == reinterpret_cast<ur_device_handle_t>(1)
                ? UR_DEVICE_TYPE_CPU
                : UR_DEVICE_TYPE_GPU;
      if (*P.ppropName == UR_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT &&
          *P.phDevice == reinterpret_cast<ur_device_handle_t>(2))
        *static_cast<ur_device_usm_access_capability_flags_t *>(
            *P.ppPropValue) = 0;
    }
    return UR_RESULT_SUCCESS;
  });

  const platform Plt;
  const auto Devices = Plt.get_devices();
  ASSERT_EQ(Devices.size(), 2u);
  EXPECT_TRUE(Devices[0].has(aspect::cpu));
  EXPECT_TRUE(Devices[1].has(aspect::gpu));
  EXPECT_FALSE(Plt.has(aspect::cpu));
  EXPECT_FALSE(Plt.has(aspect::gpu));
  EXPECT_TRUE(Devices[0].has(aspect::usm_shared_allocations));
  EXPECT_FALSE(Devices[1].has(aspect::usm_shared_allocations));
  EXPECT_FALSE(Plt.has(aspect::usm_shared_allocations));
  EXPECT_TRUE(Devices[0].has(aspect::fp64));
  EXPECT_TRUE(Devices[1].has(aspect::fp64));
  EXPECT_TRUE(Plt.has(aspect::fp64));
}
