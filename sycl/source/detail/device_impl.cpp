//==----------------- device_impl.cpp - SYCL device ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/jit_compiler.hpp>
#include <detail/platform_impl.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/ur_info_code.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {
namespace detail {

/// Constructs a SYCL device instance using the provided
/// UR device instance.
device_impl::device_impl(ur_device_handle_t Device, platform_impl &Platform,
                         device_impl::private_tag, size_t idx)
    : MDevice(Device), MPlatform(Platform),
      // No need to set MRootDevice when MAlwaysRootDevice is true
      MRootDevice(Platform.MAlwaysRootDevice
                      ? nullptr
                      : get_info_impl<UR_DEVICE_INFO_PARENT_DEVICE>()),
      // TODO catch an exception and put it to list of asynchronous exceptions:
      MCache{*this}, MIndexWithinPlatform(idx) {
  // Interoperability Constructor already calls DeviceRetain in
  // urDeviceCreateWithNativeHandle.
  getAdapter().call<UrApiKind::urDeviceRetain>(MDevice);
}

device_impl::~device_impl() {
  try {
    // TODO catch an exception and put it to list of asynchronous exceptions
    adapter_impl &Adapter = getAdapter();
    ur_result_t Err = Adapter.call_nocheck<UrApiKind::urDeviceRelease>(MDevice);
    __SYCL_CHECK_UR_CODE_NO_EXC(Err, Adapter.getBackend());
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~device_impl", e);
  }
}

bool device_impl::is_affinity_supported(
    info::partition_affinity_domain AffinityDomain) const {
  auto SupportedDomains = get_info<info::device::partition_affinity_domains>();
  return std::find(SupportedDomains.begin(), SupportedDomains.end(),
                   AffinityDomain) != SupportedDomains.end();
}

cl_device_id device_impl::get() const {
  // TODO catch an exception and put it to list of asynchronous exceptions
  __SYCL_OCL_CALL(clRetainDevice, ur::cast<cl_device_id>(getNative()));
  return ur::cast<cl_device_id>(getNative());
}

platform device_impl::get_platform() const {
  return createSyclObjFromImpl<platform>(MPlatform);
}

bool device_impl::has_extension(const std::string &ExtensionName) const {
  if (ExtensionName.empty())
    return false;

  const std::string AllExtensionNames{
      get_info_impl<UR_DEVICE_INFO_EXTENSIONS>()};

  size_t FoundExtPos = AllExtensionNames.find(ExtensionName);
  while (FoundExtPos != std::string::npos) {
    // If the extension name was found, we need to ensure it is not a partial
    // match. That is, the following must hold:
    //  * The match must be at the start of the list of names or have a
    //    whitespace before it and
    //  * the match must end at the end of the list of names or have a
    //    whitespace after it.
    bool IsStartOrTerminated =
        FoundExtPos == 0 || AllExtensionNames[FoundExtPos - 1] == ' ';
    bool IsEndOrTerminated =
        FoundExtPos + ExtensionName.size() == AllExtensionNames.size() ||
        AllExtensionNames[FoundExtPos + ExtensionName.size()] == ' ';
    if (IsStartOrTerminated && IsEndOrTerminated)
      return true;

    // If the match was partial, the extension name could still be later in the
    // list. As such, search for the next match and recheck.
    FoundExtPos = AllExtensionNames.find(ExtensionName,
                                         FoundExtPos + ExtensionName.size());
  }
  return false;
}

bool device_impl::is_partition_supported(info::partition_property Prop) const {
  auto SupportedProperties = get_info<info::device::partition_properties>();
  return std::find(SupportedProperties.begin(), SupportedProperties.end(),
                   Prop) != SupportedProperties.end();
}

std::vector<device> device_impl::create_sub_devices(
    const ur_device_partition_properties_t *Properties,
    size_t SubDevicesCount) const {
  std::vector<ur_device_handle_t> SubDevices(SubDevicesCount);
  uint32_t ReturnedSubDevices = 0;
  adapter_impl &Adapter = getAdapter();
  Adapter.call<sycl::errc::invalid, UrApiKind::urDevicePartition>(
      MDevice, Properties, SubDevicesCount, SubDevices.data(),
      &ReturnedSubDevices);
  if (ReturnedSubDevices != SubDevicesCount) {
    throw sycl::exception(
        errc::invalid,
        "Could not partition to the specified number of sub-devices");
  }
  // TODO: Need to describe the subdevice model. Some sub_device management
  // may be necessary. What happens if create_sub_devices is called multiple
  // times with the same arguments?
  //
  std::vector<device> res;
  std::for_each(SubDevices.begin(), SubDevices.end(),
                [&res, this](const ur_device_handle_t &a_ur_device) {
                  device sycl_device = detail::createSyclObjFromImpl<device>(
                      MPlatform.getOrMakeDeviceImpl(a_ur_device));
                  res.push_back(sycl_device);
                });
  // urDevicePartition returns devices with their reference counts
  // incremented. Each device_impl wrapper increments the reference count and
  // decrements it on destruction (shared ownership). So, we have to decrement
  // the reference count once here to release temporary handles.
#ifdef _WIN32
  // On Windows OpenCL backend, releasing the sub-devices here leads to a crash
  // during late shutdown. There have been issues observed with premature
  // unloading of opencl related dlls and seems like that might be the case. So,
  // intentionally leak sub-devices on Windows OpenCL backend for now.
  // TODO: Remove this workaround.
  if (getAdapter().getBackend() != backend::opencl)
#endif
    for (ur_device_handle_t &SubDevice : SubDevices)
      Adapter.call<UrApiKind::urDeviceRelease>(SubDevice);

  return res;
}

std::vector<device> device_impl::create_sub_devices(size_t ComputeUnits) const {
  if (!is_partition_supported(info::partition_property::partition_equally)) {
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Device does not support "
                          "sycl::info::partition_property::partition_equally.");
  }
  // If count exceeds the total number of compute units in the device, an
  // exception with the errc::invalid error code must be thrown.
  auto MaxComputeUnits = get_info<info::device::max_compute_units>();
  if (ComputeUnits > MaxComputeUnits)
    throw sycl::exception(errc::invalid,
                          "Total counts exceed max compute units");

  size_t SubDevicesCount = MaxComputeUnits / ComputeUnits;

  ur_device_partition_property_t Prop{};
  Prop.type = UR_DEVICE_PARTITION_EQUALLY;
  Prop.value.count = static_cast<uint32_t>(ComputeUnits);

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.PropCount = 1;
  Properties.pProperties = &Prop;

  return create_sub_devices(&Properties, SubDevicesCount);
}

std::vector<device>
device_impl::create_sub_devices(const std::vector<size_t> &Counts) const {
  if (!is_partition_supported(info::partition_property::partition_by_counts)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::partition_by_counts.");
  }

  std::vector<ur_device_partition_property_t> Props{};

  // Fill the properties vector with counts and validate it
  size_t TotalCounts = 0;
  size_t NonZeroCounts = 0;
  for (auto Count : Counts) {
    TotalCounts += Count;
    NonZeroCounts += (Count != 0) ? 1 : 0;
    Props.push_back(ur_device_partition_property_t{
        UR_DEVICE_PARTITION_BY_COUNTS, {static_cast<uint32_t>(Count)}});
  }

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.pProperties = Props.data();
  Properties.PropCount = Props.size();

  // If the number of non-zero values in counts exceeds the device’s maximum
  // number of sub devices (as returned by info::device::
  // partition_max_sub_devices) an exception with the errc::invalid
  // error code must be thrown.
  if (NonZeroCounts > get_info<info::device::partition_max_sub_devices>())
    throw sycl::exception(errc::invalid,
                          "Total non-zero counts exceed max sub-devices");

  // If the total of all the values in the counts vector exceeds the total
  // number of compute units in the device (as returned by
  // info::device::max_compute_units), an exception with the errc::invalid
  // error code must be thrown.
  if (TotalCounts > get_info<info::device::max_compute_units>())
    throw sycl::exception(errc::invalid,
                          "Total counts exceed max compute units");

  return create_sub_devices(&Properties, Counts.size());
}

static inline std::string
affinityDomainToString(info::partition_affinity_domain AffinityDomain) {
  switch (AffinityDomain) {
#define __SYCL_AFFINITY_DOMAIN_STRING_CASE(DOMAIN)                             \
  case DOMAIN:                                                                 \
    return #DOMAIN;

    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::numa)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L4_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L3_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L2_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::L1_cache)
    __SYCL_AFFINITY_DOMAIN_STRING_CASE(
        sycl::info::partition_affinity_domain::next_partitionable)
#undef __SYCL_AFFINITY_DOMAIN_STRING_CASE
  default:
    assert(false && "Missing case for affinity domain.");
    return "unknown";
  }
}

std::vector<device> device_impl::create_sub_devices(
    info::partition_affinity_domain AffinityDomain) const {
  if (!is_partition_supported(
          info::partition_property::partition_by_affinity_domain)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::partition_by_affinity_domain.");
  }
  if (!is_affinity_supported(AffinityDomain)) {
    throw sycl::exception(make_error_code(errc::feature_not_supported),
                          "Device does not support " +
                              affinityDomainToString(AffinityDomain) + ".");
  }

  ur_device_partition_property_t Prop{};
  Prop.type = UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN;
  Prop.value.affinity_domain =
      static_cast<ur_device_affinity_domain_flags_t>(AffinityDomain);

  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.PropCount = 1;
  Properties.pProperties = &Prop;

  uint32_t SubDevicesCount = 0;
  adapter_impl &Adapter = getAdapter();
  Adapter.call<sycl::errc::invalid, UrApiKind::urDevicePartition>(
      MDevice, &Properties, 0u, nullptr, &SubDevicesCount);

  return create_sub_devices(&Properties, SubDevicesCount);
}

std::vector<device> device_impl::create_sub_devices() const {
  if (!is_partition_supported(
          info::partition_property::ext_intel_partition_by_cslice)) {
    throw sycl::exception(
        make_error_code(errc::feature_not_supported),
        "Device does not support "
        "sycl::info::partition_property::ext_intel_partition_by_cslice.");
  }

  ur_device_partition_property_t Prop{};
  Prop.type = UR_DEVICE_PARTITION_BY_CSLICE;
  ur_device_partition_properties_t Properties{};
  Properties.stype = UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES;
  Properties.pProperties = &Prop;
  Properties.PropCount = 1;

  uint32_t SubDevicesCount = 0;
  adapter_impl &Adapter = getAdapter();
  Adapter.call<UrApiKind::urDevicePartition>(MDevice, &Properties, 0u, nullptr,
                                             &SubDevicesCount);

  return create_sub_devices(&Properties, SubDevicesCount);
}

ur_native_handle_t device_impl::getNative() const {
  adapter_impl &Adapter = getAdapter();
  ur_native_handle_t Handle;
  Adapter.call<UrApiKind::urDeviceGetNativeHandle>(getHandleRef(), &Handle);
  if (getBackend() == backend::opencl) {
    __SYCL_OCL_CALL(clRetainDevice, ur::cast<cl_device_id>(Handle));
  }
  return Handle;
}

// Adaptive timestamp caching mechanism to reduce expensive calls.
//
// This implementation uses an adaptive strategy:
// - Caches the last actual GPU/CPU timestamp pair
// - Estimates GPU time based on CPU elapsed time to avoid expensive backend calls
// - Dynamically adjusts refresh interval (1ms-1s) based on prediction accuracy
// - Refreshes when CPU elapsed time exceeds current timeout
// - If prediction error > 5%, decreases timeout (more frequent refreshes)
// - If prediction error <= 5%, increases timeout (less frequent refreshes)
//
uint64_t device_impl::getCurrentDeviceTime() {
  auto GetGlobalTimestamps = [this](ur_device_handle_t Device,
                                    uint64_t *DeviceTime, uint64_t *HostTime) {
    auto Result =
        getAdapter().call_nocheck<UrApiKind::urDeviceGetGlobalTimestamps>(
            Device, DeviceTime, HostTime);
    if (Result == UR_RESULT_ERROR_INVALID_OPERATION) {
      // NOTE(UR port): Removed the call to GetLastError because  we shouldn't
      // be calling it after ERROR_INVALID_OPERATION: there is no
      // adapter-specific error.
      throw detail::set_ur_error(
          sycl::exception(
              make_error_code(errc::feature_not_supported),
              "Device and/or backend does not support querying timestamp."),
          UR_RESULT_ERROR_INVALID_OPERATION);
    } else {
      getAdapter().checkUrResult<errc::feature_not_supported>(Result);
    }
  };

  std::unique_lock<std::shared_mutex> WriteLock(MDeviceHostBaseTimeMutex);

  // Get current CPU time
  uint64_t CurrentHostTime = 0;
  GetGlobalTimestamps(MDevice, nullptr, &CurrentHostTime);

  // Determine if we need to refresh from KMD
  bool RefreshTimestamps = false;
  if (!MDeviceHostBaseTime.second) {
    // First call - must query actual timestamps
    RefreshTimestamps = true;
  } else if (MTimestampActualCallCount < MTimestampCallsBeforeAdaptation) {
    // Warmup phase: always query actual timestamps for first few calls
    // to establish a reliable baseline before enabling adaptive logic
    RefreshTimestamps = true;
  } else {
    // Calculate CPU time elapsed since last actual query
    // Handle clock backward jump or wrap
    if (CurrentHostTime < MDeviceHostBaseTime.second) {
      // Clock went backwards - force refresh to re-establish base
      RefreshTimestamps = true;
    } else {
      uint64_t CpuTimeDiffInNS = CurrentHostTime - MDeviceHostBaseTime.second;
      if (CpuTimeDiffInNS >= MTimestampRefreshTimeoutNS) {
        RefreshTimestamps = true;
      }
    }
  }

  if (RefreshTimestamps) {
    std::cout << "Refresh" << std::endl;
    // Query actual GPU and CPU timestamps from KMD
    uint64_t ActualDeviceTime = 0;
    uint64_t ActualHostTime = 0;
    GetGlobalTimestamps(MDevice, &ActualDeviceTime, &ActualHostTime);

    // Adaptive timeout adjustment (skip during warmup phase)
    if (MTimestampActualCallCount >= MTimestampCallsBeforeAdaptation) {
      // Check for clock backward jumps or wraps (using OLD cached values)
      bool ClockAnomalyDetected = false;
      if (ActualHostTime < MDeviceHostBaseTime.second ||
          ActualDeviceTime < MDeviceHostBaseTime.first) {
        // Clock went backwards - skip adaptive logic, just reset base
        ClockAnomalyDetected = true;
      }

      if (!ClockAnomalyDetected) {
        // Calculate what the GPU timestamp would have been if we estimated it
        // Compute differences in signed space to avoid unsigned wrap
        int64_t CpuTimeDiff = static_cast<int64_t>(ActualHostTime) -
                               static_cast<int64_t>(MDeviceHostBaseTime.second);
        int64_t GpuTimeDiff = static_cast<int64_t>(ActualDeviceTime) -
                               static_cast<int64_t>(MDeviceHostBaseTime.first);

        // Only proceed if differences are positive (monotonic)
        if (CpuTimeDiff > 0 && GpuTimeDiff > 0) {
            // Update clock ratio: how much GPU time advances per CPU time
            // Use exponential moving average to smooth out noise
            double ObservedRatio = static_cast<double>(GpuTimeDiff) /
                                    static_cast<double>(CpuTimeDiff);
            constexpr double Alpha = 0.3; // smoothing factor
            double OldRatio = MDeviceHostClockRatio;
            MDeviceHostClockRatio = Alpha * ObservedRatio +
                                     (1.0 - Alpha) * OldRatio;

            // Calculate what we would have estimated (using OLD ratio)
            uint64_t CalculatedDeviceTime = MDeviceHostBaseTime.first +
                                             static_cast<uint64_t>(static_cast<double>(CpuTimeDiff) * OldRatio);

            // Calculate absolute prediction error robustly
            uint64_t DiffAbs = (ActualDeviceTime > CalculatedDeviceTime)
                                   ? (ActualDeviceTime - CalculatedDeviceTime)
                                   : (CalculatedDeviceTime - ActualDeviceTime);

            // Track maximum observed absolute error for safety margin calculation
            if (DiffAbs > MMaxObservedError) {
              MMaxObservedError = DiffAbs;
            }

            uint64_t ElapsedNS = static_cast<uint64_t>(GpuTimeDiff);

            // Require at least 1ms elapsed time to avoid noisy ratios
            constexpr uint64_t MinElapsedForAdaptation = 1000000ULL; // 1ms
            if (ElapsedNS > MinElapsedForAdaptation) {
              // Use relative error to decide adaptation (2% threshold)
              double RelError = static_cast<double>(DiffAbs) / static_cast<double>(ElapsedNS);
              if (RelError > 0.02) {
                // High relative error: decrease timeout (more frequent refreshes)
                MTimestampRefreshTimeoutNS = std::max(MTimestampRefreshMinTimeoutNS,
                                                      MTimestampRefreshTimeoutNS / 2);
              } else {
                // Low error: increase timeout (less frequent refreshes)
                // Be conservative: multiply by 2
                uint64_t NewTimeout = MTimestampRefreshTimeoutNS * 2;
                MTimestampRefreshTimeoutNS = std::min(MTimestampRefreshMaxTimeoutNS, NewTimeout);
              }
            }
        }
      }
    }

    // Update cached timestamps after adaptive logic
    MDeviceHostBaseTime.first = ActualDeviceTime;
    MDeviceHostBaseTime.second = ActualHostTime;
    MTimestampActualCallCount++;
    MLastReturnedDeviceTime = ActualDeviceTime;

    return ActualDeviceTime;
  } else {
        std::cout << "Estimate" << std::endl;

    // Use cached timestamps and estimate GPU time from CPU elapsed time
    // Apply clock ratio correction and conservative safety margin
    uint64_t CpuTimeDiffInNS = CurrentHostTime - MDeviceHostBaseTime.second;

    // Apply observed clock ratio to get better estimate
    double EstimatedGpuDiff = static_cast<double>(CpuTimeDiffInNS) *
                               MDeviceHostClockRatio;

    // Apply conservative safety margin based on worst-case observed error
    // Subtract max observed error plus additional 10% buffer to ensure
    // we never go ahead of actual GPU time
    int64_t SafetyMargin = MMaxObservedError +
                            static_cast<int64_t>(EstimatedGpuDiff * 0.1);
    // Cap safety margin to reasonable bounds
    SafetyMargin = std::min(SafetyMargin, static_cast<int64_t>(CpuTimeDiffInNS / 2));
    SafetyMargin = std::max(SafetyMargin, static_cast<int64_t>(0));

    int64_t SafeEstimate = static_cast<int64_t>(EstimatedGpuDiff) - SafetyMargin;
    SafeEstimate = std::max(SafeEstimate, static_cast<int64_t>(0));

    uint64_t EstimatedDeviceTime = MDeviceHostBaseTime.first +
                                    static_cast<uint64_t>(SafeEstimate);

    // Enforce monotonicity: never return a timestamp earlier than before
    if (EstimatedDeviceTime < MLastReturnedDeviceTime) {
      EstimatedDeviceTime = MLastReturnedDeviceTime;
    }

    MLastReturnedDeviceTime = EstimatedDeviceTime;
    return EstimatedDeviceTime;
  }
}

bool device_impl::extOneapiCanBuild(
    ext::oneapi::experimental::source_language Language) {
  try {
    return sycl::ext::oneapi::experimental::detail::
        is_source_kernel_bundle_supported(Language,
                                          std::vector<device_impl *>{this});

  } catch (sycl::exception &) {
    return false;
  }
}

bool device_impl::extOneapiCanCompile(
    ext::oneapi::experimental::source_language Language) {
  try {
    // Currently only SYCL language is supported for compiling.
    return Language == ext::oneapi::experimental::source_language::sycl &&
           sycl::ext::oneapi::experimental::detail::
               is_source_kernel_bundle_supported(
                   Language, std::vector<device_impl *>{this});
  } catch (sycl::exception &) {
    return false;
  }
}

// Returns the strongest guarantee that can be provided by the host device for
// threads created at threadScope from a coordination scope given by
// coordinationScope
sycl::ext::oneapi::experimental::forward_progress_guarantee
device_impl::getHostProgressGuarantee(
    ext::oneapi::experimental::execution_scope,
    ext::oneapi::experimental::execution_scope) {
  return sycl::ext::oneapi::experimental::forward_progress_guarantee::
      weakly_parallel;
}

// Returns the strongest progress guarantee that can be provided by this device
// for threads created at threadScope from the coordination scope given by
// coordinationScope.
sycl::ext::oneapi::experimental::forward_progress_guarantee
device_impl::getProgressGuarantee(
    ext::oneapi::experimental::execution_scope threadScope,
    ext::oneapi::experimental::execution_scope coordinationScope) const {
  using forward_progress_guarantee =
      ext::oneapi::experimental::forward_progress_guarantee;
  using execution_scope = ext::oneapi::experimental::execution_scope;
  const int executionScopeSize = 4;
  (void)coordinationScope;
  int threadScopeNum = static_cast<int>(threadScope);
  // we get the immediate progress guarantee that is provided by each scope
  // between root_group and threadScope and then return the weakest of these.
  // Counterintuitively, this corresponds to taking the max of the enum values
  // because of how the forward_progress_guarantee enum values are declared.
  int guaranteeNum = static_cast<int>(
      getImmediateProgressGuarantee(execution_scope::root_group));
  for (int currentScope = executionScopeSize - 2; currentScope > threadScopeNum;
       --currentScope) {
    guaranteeNum = std::max(guaranteeNum,
                            static_cast<int>(getImmediateProgressGuarantee(
                                static_cast<execution_scope>(currentScope))));
  }
  return static_cast<forward_progress_guarantee>(guaranteeNum);
}

bool device_impl::supportsForwardProgress(
    ext::oneapi::experimental::forward_progress_guarantee guarantee,
    ext::oneapi::experimental::execution_scope threadScope,
    ext::oneapi::experimental::execution_scope coordinationScope) const {
  auto guarantees = getProgressGuaranteesUpTo(
      getProgressGuarantee(threadScope, coordinationScope));
  return std::find(guarantees.begin(), guarantees.end(), guarantee) !=
         guarantees.end();
}

// Returns the progress guarantee provided for a coordination scope
// given by coordination_scope for threads created at a scope
// immediately below coordination_scope. For example, for root_group
// coordination scope it returns the progress guarantee provided
// at root_group for threads created at work_group.
ext::oneapi::experimental::forward_progress_guarantee
device_impl::getImmediateProgressGuarantee(
    ext::oneapi::experimental::execution_scope coordination_scope) const {
  using forward_progress_guarantee =
      ext::oneapi::experimental::forward_progress_guarantee;
  using execution_scope = ext::oneapi::experimental::execution_scope;
  if (is_cpu() && getBackend() == backend::opencl) {
    switch (coordination_scope) {
    case execution_scope::root_group:
      return forward_progress_guarantee::parallel;
    case execution_scope::work_group:
    case execution_scope::sub_group:
      return forward_progress_guarantee::weakly_parallel;
    default:
      throw sycl::exception(sycl::errc::invalid,
                            "Work item is not a valid coordination scope!");
    }
  } else if (is_gpu() && getBackend() == backend::ext_oneapi_level_zero) {
    switch (coordination_scope) {
    case execution_scope::root_group:
    case execution_scope::work_group:
      return forward_progress_guarantee::concurrent;
    case execution_scope::sub_group:
      return forward_progress_guarantee::weakly_parallel;
    default:
      throw sycl::exception(sycl::errc::invalid,
                            "Work item is not a valid coordination scope!");
    }
  }
  return forward_progress_guarantee::weakly_parallel;
}

void device_impl::wait() {
  // Firstly, all associated queues should be cleaned through of all
  // not-yet-enqueued commands and host_task.
  {
    std::lock_guard<std::mutex> Lock(MQueuesMutex);
    for (const std::weak_ptr<queue_impl> &WQueue : MQueues) {
      std::shared_ptr<queue_impl> Queue = WQueue.lock();
      assert(Queue && "Queue should never be dangling in the list of queues "
                      "associated with the device!");
      Queue->waitForRuntimeLevelCmdsAndClear();
    }
  }

  // Then we synchronize the entire device.
  getAdapter().call<detail::UrApiKind::urDeviceWaitExp>(getHandleRef());
}

void device_impl::throwAsynchronous() {
  Scheduler::getInstance().flushAsyncExceptions();
}

} // namespace detail
} // namespace _V1
} // namespace sycl
