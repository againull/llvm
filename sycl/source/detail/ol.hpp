//==---------- ol.hpp - Liboffload integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// C++ utilities for Liboffload integration.
///
/// \ingroup sycl_ol

#pragma once

#include <detail/offload_dispatcher.hpp>
#include <detail/offload_topology.hpp>
#include <memory>
#include <offload/OffloadAPI.h>
#include <string>
#include <vector>

namespace sycl {
inline namespace _V1 {
enum class backend : char;
namespace detail {
class adapter_impl;

namespace ol {
void *getLiboffloadLibrary();

OffloadLib &initializeLibOffload();

// Get the adapter serving given backend.
template <backend BE> adapter_impl &getAdapter();

OffloadLib &getOffloadLib();

// Get the topology for given backend.
OffloadTopology &getOffloadTopology(ol_platform_backend_t BE);
} // namespace ol

// Convert from Liboffload backend to SYCL backend enum
backend convertOlBackend(ol_platform_backend_t OlBackend);

template <auto ApiKind, typename SyclImplTy, typename DescTy>
std::string olGetInfoString(SyclImplTy &SyclImpl, DescTy Desc) {
  // Avoid explicit type to keep template-type-dependent.
  auto &Offload = ol::getOffloadLib();
  size_t ResultSize = 0;
  auto Handle = SyclImpl.getOlHandleRef();
  Offload.template call<ApiKind>(Handle, Desc,
                                 /*propSize=*/0,
                                 /*pPropValue=*/nullptr, &ResultSize);
  if (ResultSize == 0)
    return std::string{};

  std::string Result;
  // C++23's `resize_and_overwrite` would be better...
  //
  // Liboffload counts null terminator in the size, std::string doesn't. Adjust
  // by "-1" for that.
  Result.resize(ResultSize - 1);
  Offload.template call<ApiKind>(Handle, Desc, ResultSize, Result.data(),
                                 nullptr);

  return Result;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
