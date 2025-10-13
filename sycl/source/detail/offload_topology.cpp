//===-- offload_topology.cpp - liboffload devices discovery ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "offload_topology.hpp"
#include "offload_dispatcher.hpp"
#include <detail/global_handler.hpp>
#include <mutex>
#include <sycl/detail/ol.hpp>
#include <unordered_map>

namespace sycl {
inline namespace _V1 {
namespace detail {

void discoverOflloadDevices(OffloadDispatcher &Dispatcher) {
  static std::once_flag DiscoverOnce;
  std::call_once(DiscoverOnce, [&]() {
    std::array<std::unordered_map<ol_platform_handle_t,
                                  std::vector<ol_device_handle_t>>,
               OL_PLATFORM_BACKEND_LAST>
        Mapping;
    struct CBData {
      OffloadDispatcher *Dispatcher;
      decltype(Mapping) *MappingPtr;
    } CB{&Dispatcher, &Mapping};
    Dispatcher.call_nocheck<OlApiKind::olIterateDevices>(
        [](ol_device_handle_t Dev, void *User) -> bool {
          auto *D = static_cast<CBData *>(User);
          ol_platform_handle_t Plat = nullptr;
          ol_result_t Res =
              D->Dispatcher->call_nocheck<OlApiKind::olGetDeviceInfo>(
                  Dev, OL_DEVICE_INFO_PLATFORM, sizeof(Plat), &Plat);
          if (Res != OL_SUCCESS)
            return true; // continue

          ol_platform_backend_t OlBackend = OL_PLATFORM_BACKEND_UNKNOWN;
          Res = D->Dispatcher->call_nocheck<OlApiKind::olGetPlatformInfo>(
              Plat, OL_PLATFORM_INFO_BACKEND, sizeof(OlBackend), &OlBackend);
          if (Res != OL_SUCCESS)
            return true; // continue

          if (OL_PLATFORM_BACKEND_HOST == OlBackend ||
              OL_PLATFORM_BACKEND_UNKNOWN == OlBackend)
            return true; // skip host/unknown backend

          // TODO: skip banned platforms

          // Ensure backend index fits into array size
          if (OlBackend >= OL_PLATFORM_BACKEND_LAST)
            return true;

          (*D->MappingPtr)[static_cast<size_t>(OlBackend)][Plat].push_back(Dev);
          return true;
        },
        &CB);
    // Now register all platforms and devices into the topologies
    auto &BackendTopologies = GlobalHandler::instance().getOffloadTopologies();
    for (size_t I = 0; I < OL_PLATFORM_BACKEND_LAST; ++I) {
      Topology &Topo = BackendTopologies[I];
      Topo.set_backend(static_cast<ol_platform_backend_t>(I));
      for (auto &PltAndDevs : Mapping[I])
        Topo.register_new_platform_and_devices(PltAndDevs.first,
                                               std::move(PltAndDevs.second));
    }
  });
}

} // namespace detail
} // namespace _V1
} // namespace sycl
