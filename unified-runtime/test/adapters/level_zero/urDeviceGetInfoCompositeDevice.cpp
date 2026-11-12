// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %maybe-v1 ./root_device-test
// RUN: %maybe-v2 ./root_device-test

#include "ze_api.h"
#include <uur/fixtures.h>

using urLevelZeroRootDeviceTest = uur::urDeviceTest;
UUR_INSTANTIATE_DEVICE_TEST_SUITE(urLevelZeroRootDeviceTest);

TEST_P(urLevelZeroRootDeviceTest, RepeatedCompositeQueriesOnLegacyDriver) {
  ASSERT_EQ(ZE_RESULT_SUCCESS, zeInit(ZE_INIT_FLAG_GPU_ONLY));
  ur_platform_handle_t platform = nullptr;
  ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_PLATFORM,
                                 sizeof(platform), &platform, nullptr));
  ur_native_handle_t nativePlatform = 0;
  ASSERT_SUCCESS(urPlatformGetNativeHandle(platform, &nativePlatform));
  ze_api_version_t version = ZE_API_VERSION_CURRENT;
  ASSERT_EQ(
      ZE_RESULT_SUCCESS,
      zeDriverGetApiVersion(
          reinterpret_cast<ze_driver_handle_t>(nativePlatform), &version));

  for (unsigned call = 0; call != 2; ++call) {
    ur_device_handle_t composite = device;
    ASSERT_SUCCESS(urDeviceGetInfo(device, UR_DEVICE_INFO_COMPOSITE_DEVICE,
                                   sizeof(composite), &composite, nullptr));
    if (version < ZE_API_VERSION_1_7) {
      EXPECT_EQ(composite, nullptr);
    }
  }
}
