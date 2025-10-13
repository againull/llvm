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
#include <cassert>
#include <iostream>
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

struct Range {
  uint32_t begin = 0, count = 0;
};

// Contiguous global storage of platform and device handles for a backend.
struct Topology {
  Topology() : OlBackend(OL_PLATFORM_BACKEND_UNKNOWN) {}
  Topology(ol_platform_backend_t OlBackend) : OlBackend(OlBackend) {}

  void set_backend(ol_platform_backend_t B) { OlBackend = B; }

  // Platforms for this backend
  range_view<ol_platform_handle_t> platforms() const {
    return {Platforms.data(), Platforms.size()};
  }

  // Devices for a specific platform (platform_id is index into Platforms)
  range_view<ol_device_handle_t>
  devices_for_platform(size_t platform_id) const {
    if (platform_id >= PlatformDevices.size())
      return {nullptr, 0};
    const auto r = PlatformDevices[platform_id];
    return {Devices.data() + r.begin, r.count};
  }

  size_t get_first_device_index_for_platform(size_t platform_id) const {
    assert(platform_id < PlatformDevices.size());
    const auto r = PlatformDevices[platform_id];
    return r.begin;
  }

  // All devices for this backend (consecutive across platforms)
  range_view<ol_device_handle_t> devices() const {
    return {Devices.data(), Devices.size()};
  }

  // Register new platform and devices into this topology under that platform.
  void
  register_new_platform_and_devices(ol_platform_handle_t NewPlatform,
                                    std::vector<ol_device_handle_t> &&NewDevs) {
    Platforms.push_back(NewPlatform);

    Range R;
    R.begin = Devices.size();
    R.count = NewDevs.size();
    Devices.insert(Devices.end(), NewDevs.begin(), NewDevs.end());
    PlatformDevices.push_back(R);
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
};

// Initialize the topologies by calling olIterateDevices.
void discoverOflloadDevices(class OffloadDispatcher &Dispatcher);

} // namespace detail
} // namespace _V1
} // namespace sycl
