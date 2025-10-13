//==---------- ol.cpp - Liboffload integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// Implementation of C++ utilities for Liboffload integration.
///
/// \ingroup sycl_ol

#include "ol.hpp"
#include <detail/global_handler.hpp>
#include <detail/offload_dispatcher.hpp>
#include <offload/OffloadAPI.h>
#include <sycl/detail/common.hpp>
#include <sycl/detail/ol.hpp>

#include <bitset>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stddef.h>
#include <string>
#include <tuple>

namespace sycl {
inline namespace _V1 {
namespace detail {
namespace ol {

OffloadDispatcher &initializeLibOffload() {
  // This uses static variable initialization to work around a gcc bug with
  // std::call_once and exceptions.
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66146
  auto initializeHelper = [=]() {
    OlFuncInfo<OlApiKind::olInit> OffloadInitInfo;
    auto OffloadInit =
        OffloadInitInfo.getFuncPtrFromModule(ol::getLiboffloadLibrary());
    std::cout << "Initializing liboffload\n";
    ol_result_t Res = OffloadInit();
    if (Res != OL_SUCCESS) {
      std::cerr << "Liboffload initialization failed" << std::endl;
      exit(1);
    }
    return true;
  };
  static bool Initialized = initializeHelper();
  (void)Initialized;
  return GlobalHandler::instance().getOffloadDispatcher();
}

// Get the topology for the given backend.
Topology &getBackendTopology(ol_platform_backend_t BE) {
  // Topologies are indexed by ol_platform_backend_t, which matches
  // backend enum values for all supported backends.
  auto &BackendTopologies = GlobalHandler::instance().getOffloadTopologies();
  size_t BEIdx = static_cast<size_t>(BE);
  std::cout << "BEIdx: " << BEIdx << "\n";
  if (BEIdx < BackendTopologies.size() &&
      BackendTopologies[BEIdx].backend() != OL_PLATFORM_BACKEND_UNKNOWN)
    return BackendTopologies[BEIdx];

  throw exception(errc::runtime, "Couldn't find topology for backend");
}

OffloadDispatcher &getOffloadDispatcher() {
  return GlobalHandler::instance().getOffloadDispatcher();
}

} // namespace ol
} // namespace detail
} // namespace _V1
} // namespace sycl
