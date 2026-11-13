// Check work-group limit queries and launch diagnostics without executing huge
// grids on a device. Distinct limits also check the SYCL/UR dimension ordering.
#include <helpers/MockDeviceImage.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>
#include <sycl/ext/oneapi/experimental/max_work_groups.hpp>
#include <sycl/sycl.hpp>

class WorkGroupLimits1;
class WorkGroupLimits2;
class WorkGroupLimits3;
MOCK_INTEGRATION_HEADER(WorkGroupLimits1)
MOCK_INTEGRATION_HEADER(WorkGroupLimits2)
MOCK_INTEGRATION_HEADER(WorkGroupLimits3)

namespace {
sycl::unittest::MockDeviceImage Image{
    SYCL_DEVICE_BINARY_TYPE_NATIVE,
    __SYCL_DEVICE_BINARY_TARGET_NVPTX64,
    "",
    "",
    std::vector<unsigned char>{0},
    sycl::unittest::makeEmptyKernels(
        {"WorkGroupLimits1", "WorkGroupLimits2", "WorkGroupLimits3"}),
    sycl::unittest::MockPropertySet{}};
sycl::unittest::MockDeviceImageArray<1> Images{&Image};
constexpr size_t Limits[] = {7, 11, 13};

template <typename Kernel, int Dims> void checkLimits(sycl::queue &Q) {
  const auto Max = Q.get_device()
                       .get_info<sycl::ext::oneapi::experimental::info::device::
                                     max_work_groups<Dims>>();
  for (int Dim = 0; Dim < Dims; ++Dim) {
    const int URDim = Dims - Dim - 1;
    EXPECT_EQ(Max[Dim], Limits[URDim]);
    sycl::range<Dims> Local, Global;
    for (int I = 0; I < Dims; ++I)
      Local[I] = Global[I] = 1;
    Local[Dim] = 32;
    Global[Dim] = Max[Dim] * Local[Dim];
    auto Submit = [&] {
      Q.parallel_for<Kernel>(sycl::nd_range<Dims>{Global, Local},
                             [](sycl::nd_item<Dims>) {});
      Q.wait_and_throw();
    };
    EXPECT_NO_THROW(Submit());
    Global[Dim] += Local[Dim];
    try {
      Submit();
      ADD_FAILURE() << "Over-limit launch did not throw";
    } catch (const sycl::exception &E) {
      EXPECT_EQ(E.code(), sycl::errc::nd_range);
      const std::string Expected =
          "Number of work-groups exceed limit for dimension " +
          std::to_string(URDim) + " : " + std::to_string(Max[Dim] + 1) + " > " +
          std::to_string(Max[Dim]);
      EXPECT_STREQ(E.what(), Expected.c_str());
    }
  }
}
} // namespace

TEST(WorkGroupLimits, BoundaryAndExceededLimits) {
  sycl::unittest::UrMock<sycl::backend::ext_oneapi_cuda> Mock;
  mock::getCallbacks().set_replace_callback(
      "urDeviceGetInfo", [](void *Params) {
        auto &P = *static_cast<ur_device_get_info_params_t *>(Params);
        if (*P.ppropName != UR_DEVICE_INFO_MAX_WORK_GROUPS_3D)
          return sycl::unittest::MockAdapter::mock_urDeviceGetInfo(Params);
        if (*P.ppPropSizeRet)
          **P.ppPropSizeRet = sizeof(Limits);
        if (*P.ppPropValue)
          std::memcpy(*P.ppPropValue, Limits, sizeof(Limits));
        return UR_RESULT_SUCCESS;
      });
  mock::getCallbacks().set_before_callback(
      "urEnqueueKernelLaunchWithArgsExp", [](void *Params) {
        auto &P =
            *static_cast<ur_enqueue_kernel_launch_with_args_exp_params_t *>(
                Params);
        for (uint32_t I = 0; I < *P.pworkDim; ++I) {
          if ((*P.ppGlobalWorkSize)[I] / (*P.ppLocalWorkSize)[I] > Limits[I])
            return UR_RESULT_ERROR_INVALID_VALUE;
        }
        return UR_RESULT_SUCCESS;
      });
  sycl::queue Q{[](sycl::exception_list Errors) {
    for (auto Error : Errors)
      std::rethrow_exception(Error);
  }};
  checkLimits<WorkGroupLimits1, 1>(Q);
  checkLimits<WorkGroupLimits2, 2>(Q);
  checkLimits<WorkGroupLimits3, 3>(Q);
}
