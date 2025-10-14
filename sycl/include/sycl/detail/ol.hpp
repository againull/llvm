//==--------------- ol.hpp - liboffload integration helpers ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \file
///
/// C++ utilities for liboffload integration.
///
/// \ingroup sycl_offload

#pragma once

#include <offload/OffloadAPI.h>
#include <sycl/detail/export.hpp>
#include <sycl/detail/os_util.hpp>

#include <type_traits>
#include <vector>

namespace sycl {
inline namespace _V1 {
namespace detail {

enum class OlApiKind {
#define _OL_API(api) api,
#include <sycl/detail/ol_api_funcs.def>
#undef _OL_API
};

struct OlFuncPtrMapT {
#define _OL_API(api) decltype(&::api) pfn_##api = nullptr;
#include <sycl/detail/ol_api_funcs.def>
#undef _OL_API
};

template <OlApiKind OlApiOffset> struct OlFuncInfo {};

#ifdef _WIN32
void *GetWinProcAddress(void *module, const char *funcName);
inline void PopulateOlFuncPtrTable(OlFuncPtrMapT *funcs, void *module) {
#define _OL_API(api)                                                           \
  funcs->pfn_##api = (decltype(&::api))GetWinProcAddress(module, #api);
#include <sycl/detail/ol_api_funcs.def>
#undef _OL_API
}

#define _OL_API(api)                                                           \
  template <> struct OlFuncInfo<OlApiKind::api> {                              \
    using FuncPtrT = decltype(&::api);                                         \
    inline const char *getFuncName() { return #api; }                          \
    inline FuncPtrT getFuncPtr(const OlFuncPtrMapT *funcs) {                   \
      return funcs->pfn_##api;                                                 \
    }                                                                          \
    inline FuncPtrT getFuncPtrFromModule(void *module) {                       \
      return (FuncPtrT)GetWinProcAddress(module, #api);                        \
    }                                                                          \
  };
#include <sycl/detail/ol_api_funcs.def>
#undef _OL_API
#else
#define _OL_API(api)                                                           \
  template <> struct OlFuncInfo<OlApiKind::api> {                              \
    using FuncPtrT = decltype(&::api);                                         \
    inline const char *getFuncName() { return #api; }                          \
    constexpr inline FuncPtrT getFuncPtr(const void *) { return &api; }        \
    constexpr inline FuncPtrT getFuncPtrFromModule(void *) { return &api; }    \
  };
#include <sycl/detail/ol_api_funcs.def>
#undef _OL_API
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl
