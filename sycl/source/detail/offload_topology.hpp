//===--------- offload_topology.hpp - liboffload topology helper ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <algorithm>
#include <optional>
#include <sycl/detail/ol.hpp>
#include <unordered_map>

namespace sycl {
inline namespace _V1 {
namespace detail {

class OffloadDispatcher;

// Minimal span-like view
template <class T> struct range_view {
  const T *ptr{};
  size_t len{};
  const T *begin() const { return ptr; }
  const T *end() const { return ptr + len; }
  const T &operator[](size_t i) const { return ptr[i]; }
  size_t size() const { return len; }
};

using PlatformId = uint32_t;
using DeviceId = uint32_t;

struct Range {
  uint32_t begin = 0, count = 0;
};

// Contiguous global storage of platform and device handles for a backend.
struct Topology {
  Topology(ol_platform_backend_t OlBackend) : OlBackend(OlBackend) {}

  // Platforms for this backend
  range_view<ol_platform_handle_t> platforms() const {
    return {Platforms.data(), Platforms.size()};
  }

  // Devices for a specific platform (platform_id is index into Platforms)
  range_view<ol_device_handle_t>
  devices_for_platform(uint32_t platform_id) const {
    if (platform_id >= PlatformDevices.size())
      return {nullptr, 0};
    const auto r = PlatformDevices[platform_id];
    return {Devices.data() + r.begin, r.count};
  }

  range_view<ol_device_handle_t>
  devices_for_platform(ol_platform_handle_t platform) const {
    return devices_for_platform(PlatformIndex.at(platform));
  }

  // All devices for this backend (consecutive across platforms)
  range_view<ol_device_handle_t> devices() const {
    return {Devices.data(), Devices.size()};
  }

  // Map backend-local device ordinal -> handle (0..Devices.size()-1)
  ol_device_handle_t device_by_ord(uint32_t ord) const {
    if (ord >= Devices.size())
      return nullptr;
    return Devices[ord];
  }

  // Map device handle -> backend-local device ordinal (0..Devices.size()-1)
  // Linear search - only use during topology construction.
  int get_device_global_index(ol_device_handle_t H) const {
    auto It = DeviceIndex.find(H);
    if (It == DeviceIndex.end())
      return -1;
    return static_cast<int>(It->second);
  }

  std::optional<uint32_t> platform_index(ol_platform_handle_t H) const {
    auto It = PlatformIndex.find(H);
    if (It == PlatformIndex.end())
      return std::nullopt;
    return It->second;
  }

  // Register a platform and device into this topology. If the platform is
  // new, it will be added and its device range initialized. The device is
  // appended to the backend-local Devices vector and the per-platform count
  // is updated.
  void register_platform_device(ol_platform_handle_t Plt,
                                ol_device_handle_t Dev) {
    auto It = PlatformIndex.find(Plt);
    uint32_t PltIdx;
    if (It == PlatformIndex.end()) {
      PltIdx = static_cast<uint32_t>(Platforms.size());
      Platforms.push_back(Plt);
      // Device range for the new platform starts at current devices size
      PlatformDevices.push_back({static_cast<uint32_t>(Devices.size()), 0});
      PlatformIndex.emplace(Plt, PltIdx);
    } else {
      PltIdx = It->second;
    }

    uint32_t DevIdx = static_cast<uint32_t>(Devices.size());
    Devices.push_back(Dev);
    DeviceIndex.emplace(Dev, DevIdx);
    PlatformDevices[PltIdx].count++;
  }

  ol_platform_backend_t backend() { return OlBackend; }

private:
  ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;

  // Platforms and devices belonging to this backend (flattened)
  std::vector<ol_platform_handle_t> Platforms; // platforms for this backend
  std::vector<ol_device_handle_t>
      Devices; // devices for this backend (grouped by platform)

  // Vector holding range of devices for each platform (index is platform index
  // within Platforms)
  std::vector<Range>
      PlatformDevices; // PlatformDevices.size() == Platforms.size()

  // Map platform handle -> platform index
  std::unordered_map<ol_platform_handle_t, uint32_t> PlatformIndex;
  // Map device handle -> backend-local device index
  std::unordered_map<ol_device_handle_t, uint32_t> DeviceIndex;
};

// Initialize the topologies by calling olIterateDevices.
void discoverOflloadDevices(class OffloadDispatcher &Dispatcher);

} // namespace detail
} // namespace _V1
} // namespace sycl
