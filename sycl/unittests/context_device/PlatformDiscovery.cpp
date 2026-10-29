// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <helpers/ScopedEnvVar.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

namespace {

ur_result_t EnumerationError = UR_RESULT_SUCCESS;
bool FailPlatformEnumeration = false;

ur_platform_handle_t platformHandle(uintptr_t Value) {
  return reinterpret_cast<ur_platform_handle_t>(Value);
}

ur_result_t platformGet(void *Params) {
  auto &P = *static_cast<ur_platform_get_params_t *>(Params);
  const bool IsLevelZero =
      *P.phAdapter == reinterpret_cast<ur_adapter_handle_t>(1);
  if (IsLevelZero && FailPlatformEnumeration)
    return EnumerationError;
  if (*P.ppNumPlatforms)
    **P.ppNumPlatforms = 1;
  if (*P.pphPlatforms)
    **P.pphPlatforms = platformHandle(IsLevelZero ? 1 : 2);
  return UR_RESULT_SUCCESS;
}

ur_result_t platformGetInfo(void *Params) {
  auto &P = *static_cast<ur_platform_get_info_params_t *>(Params);
  if (*P.phPlatform == platformHandle(1))
    return sycl::unittest::MockAdapter::mock_urPlatformGetInfo<
        UR_BACKEND_LEVEL_ZERO>(Params);
  return sycl::unittest::MockAdapter::mock_urPlatformGetInfo<UR_BACKEND_OPENCL>(
      Params);
}

ur_result_t deviceGet(void *Params) {
  auto &P = *static_cast<ur_device_get_params_t *>(Params);
  if (*P.phPlatform == platformHandle(1))
    return EnumerationError;
  if (*P.ppNumDevices)
    **P.ppNumDevices = 1;
  if (*P.pphDevices)
    **P.pphDevices = reinterpret_cast<ur_device_handle_t>(2);
  return UR_RESULT_SUCCESS;
}

ur_result_t deviceGetInfo(void *Params) {
  auto &P = *static_cast<ur_device_get_info_params_t *>(Params);
  if (*P.ppropName == UR_DEVICE_INFO_PLATFORM) {
    if (*P.ppPropValue)
      *static_cast<ur_platform_handle_t *>(*P.ppPropValue) = platformHandle(2);
    if (*P.ppPropSizeRet)
      **P.ppPropSizeRet = sizeof(ur_platform_handle_t);
    return UR_RESULT_SUCCESS;
  }
  return sycl::unittest::MockAdapter::mock_urDeviceGetInfo(Params);
}

class PlatformDiscoveryTest : public ::testing::Test {
protected:
  void SetUp() override {
    EnumerationError = UR_RESULT_SUCCESS;
    FailPlatformEnumeration = false;
    mock::getCallbacks().set_replace_callback("urPlatformGet", platformGet);
    mock::getCallbacks().set_replace_callback("urPlatformGetInfo",
                                              platformGetInfo);
    mock::getCallbacks().set_replace_callback("urDeviceGet", deviceGet);
    mock::getCallbacks().set_replace_callback("urDeviceGetInfo", deviceGetInfo);
    auto &Adapters = sycl::detail::GlobalHandler::instance().getAdapters();
    SavedAdapters = Adapters;
    LevelZero.reset(
        new sycl::detail::adapter_impl(reinterpret_cast<ur_adapter_handle_t>(1),
                                       sycl::backend::ext_oneapi_level_zero));
    OpenCL.reset(new sycl::detail::adapter_impl(
        reinterpret_cast<ur_adapter_handle_t>(2), sycl::backend::opencl));
    Adapters = {LevelZero.get(), OpenCL.get()};
  }

  void TearDown() override {
    sycl::detail::GlobalHandler::instance().getPlatformCache().clear();
    sycl::detail::GlobalHandler::instance().getAdapters() = SavedAdapters;
  }

  sycl::unittest::UrMock<> Mock;
  std::vector<sycl::detail::adapter_impl *> SavedAdapters;
  std::unique_ptr<sycl::detail::adapter_impl> LevelZero;
  std::unique_ptr<sycl::detail::adapter_impl> OpenCL;
};

class UnavailablePlatformTest
    : public PlatformDiscoveryTest,
      public ::testing::WithParamInterface<ur_result_t> {};

TEST_P(UnavailablePlatformTest, DeviceEnumerationKeepsOtherBackends) {
  EnumerationError = GetParam();
  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1u);
  EXPECT_EQ(Platforms.front().get_backend(), sycl::backend::opencl);
  EXPECT_EQ(Platforms.front().get_devices().size(), 1u);
}

TEST_P(UnavailablePlatformTest, PlatformEnumerationKeepsOtherBackends) {
  EnumerationError = GetParam();
  FailPlatformEnumeration = true;
  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1u);
  EXPECT_EQ(Platforms.front().get_backend(), sycl::backend::opencl);
}

INSTANTIATE_TEST_SUITE_P(
    BackendUnavailable, UnavailablePlatformTest,
    ::testing::Values(UR_RESULT_ERROR_UNINITIALIZED,
                      UR_RESULT_ERROR_UNSUPPORTED_FEATURE,
                      UR_RESULT_ERROR_UNSUPPORTED_VERSION));

TEST_F(PlatformDiscoveryTest, UnexpectedErrorsAreNotHidden) {
  for (auto Error :
       {UR_RESULT_ERROR_DEVICE_LOST, UR_RESULT_ERROR_OUT_OF_HOST_MEMORY,
        UR_RESULT_ERROR_INVALID_VALUE}) {
    EnumerationError = Error;
    EXPECT_THROW(sycl::platform::get_platforms(), sycl::exception);
  }
}

TEST_F(PlatformDiscoveryTest, DirectQueriesStillReportUnavailableBackend) {
  EnumerationError = UR_RESULT_ERROR_UNINITIALIZED;
  ASSERT_EQ(sycl::platform::get_platforms().size(), 1u);
  auto Platform = sycl::detail::createSyclObjFromImpl<sycl::platform>(
      sycl::detail::platform_impl::getOrMakePlatformImpl(platformHandle(1),
                                                         *LevelZero));
  EXPECT_THROW(Platform.get_devices(), sycl::exception);
  EXPECT_THROW(sycl::device([](const sycl::device &Device) {
                 return Device.get_backend() ==
                                sycl::backend::ext_oneapi_level_zero
                            ? 1
                            : -1;
               }),
               sycl::exception);
}

TEST_F(PlatformDiscoveryTest, RuntimeWarningsExplainExclusion) {
  sycl::unittest::ScopedEnvVar Warnings(
      "SYCL_RT_WARNING_LEVEL", "1",
      sycl::detail::SYCLConfig<sycl::detail::SYCL_RT_WARNING_LEVEL>::reset);
  EnumerationError = UR_RESULT_ERROR_UNINITIALIZED;
  testing::internal::CaptureStderr();
  auto Platforms = sycl::platform::get_platforms();
  auto Diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(Platforms.size(), 1u);
  EXPECT_NE(Diagnostic.find("Skipping unavailable level_zero platform"),
            std::string::npos);
  EXPECT_NE(Diagnostic.find("UR_RESULT_ERROR_UNINITIALIZED"),
            std::string::npos);
}

TEST_F(PlatformDiscoveryTest, SkippedPlatformPreservesSelectorDeviceIndices) {
  sycl::detail::GlobalHandler::instance().getAdapters() = {LevelZero.get()};
  mock::getCallbacks().set_replace_callback("urPlatformGet", [](void *Params) {
    auto &P = *static_cast<ur_platform_get_params_t *>(Params);
    if (*P.ppNumPlatforms)
      **P.ppNumPlatforms = 3;
    if (*P.pphPlatforms)
      for (uintptr_t I = 0; I < 3; ++I)
        (*P.pphPlatforms)[I] = platformHandle(I + 1);
    return UR_RESULT_SUCCESS;
  });
  mock::getCallbacks().set_replace_callback(
      "urPlatformGetInfo", sycl::unittest::MockAdapter::mock_urPlatformGetInfo<
                               UR_BACKEND_LEVEL_ZERO>);
  mock::getCallbacks().set_replace_callback("urDeviceGet", [](void *Params) {
    auto &P = *static_cast<ur_device_get_params_t *>(Params);
    if (*P.phPlatform == platformHandle(2))
      return UR_RESULT_ERROR_UNINITIALIZED;
    if (*P.ppNumDevices)
      **P.ppNumDevices = 1;
    if (*P.pphDevices)
      **P.pphDevices = reinterpret_cast<ur_device_handle_t>(*P.phPlatform);
    return UR_RESULT_SUCCESS;
  });
  mock::getCallbacks().set_replace_callback(
      "urDeviceGetInfo", [](void *Params) {
        auto &P = *static_cast<ur_device_get_info_params_t *>(Params);
        if (*P.ppropName == UR_DEVICE_INFO_PLATFORM) {
          if (*P.ppPropValue)
            *static_cast<ur_platform_handle_t *>(*P.ppPropValue) =
                reinterpret_cast<ur_platform_handle_t>(*P.phDevice);
          if (*P.ppPropSizeRet)
            **P.ppPropSizeRet = sizeof(ur_platform_handle_t);
          return UR_RESULT_SUCCESS;
        }
        return sycl::unittest::MockAdapter::mock_urDeviceGetInfo(Params);
      });
  // Model the device index already assigned by selector filtering on the
  // first platform. The unavailable middle platform must carry it forward.
  LevelZero->getUrPlatforms();
  LevelZero->setLastDeviceId(platformHandle(1), 1);
  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 2u);
  EXPECT_EQ(sycl::detail::getSyclObjImpl(Platforms.back())->getHandleRef(),
            platformHandle(3));
  EXPECT_EQ(LevelZero->getStartingDeviceId(platformHandle(3)), 1);
}

} // namespace
