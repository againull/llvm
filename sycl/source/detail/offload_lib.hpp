#pragma once

#include <detail/config.hpp>
#include <sycl/backend_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/iostream_proxy.hpp>
#include <sycl/detail/ol.hpp>
#include <sycl/detail/type_traits.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

class OffloadLib {
public:
  OffloadLib() { populate(); }

  // Populate the dispatch table (nop for compile-time
  // binding)
  void populate() {
#ifdef _WIN32
    OlLoaderHandle = ol::getLiboffloadLibrary();
    PopulateUrFuncPtrTable(&OlFuncPtrs, OlLoaderHandle);
#endif
  }

  /// \throw SYCL 2020 exception(errc) if ol_result is not OL_SUCCESS
  template <sycl::errc errc = sycl::errc::runtime>
  void checkOlResult(ol_result_t ol_result) const {
    if (ol_result != OL_SUCCESS) {
      throw sycl::detail::set_ur_error(
          sycl::exception(sycl::make_error_code(errc),
                          sycl::detail::codeToString(ol_result->Code)),
          ol_result->Code);
    }
  }

  /// Calls the Offload Api, traces the call, and returns the result.
  ///
  /// Usage:
  /// \code{cpp}
  /// ol_result_t Err = Offload->call<OlApiKind::olEntryPoint>(Args);
  /// Offload->checkOlResult(Err); // Checks Result and throws a runtime_error
  /// // exception.
  /// \endcode
  ///
  /// \sa adapter::checkOlResult
  template <OlApiKind OlApiOffset, typename... ArgsT>
  ol_result_t call_nocheck(ArgsT &&...Args) const {
    detail::OlFuncInfo<OlApiOffset> OlApiInfo;
    auto F = OlApiInfo.getFuncPtr(&OlFuncPtrs);
    return F(std::forward<ArgsT>(Args)...);
  }

  /// Calls the API, traces the call, checks the result
  ///
  /// \throw sycl::runtime_exception if the call was not successful.
  template <OlApiKind OlApiOffset, typename... ArgsT>
  void call(ArgsT &&...Args) const {
    auto Err = call_nocheck<OlApiOffset>(std::forward<ArgsT>(Args)...);
    checkOlResult(Err);
  }

  /// \throw sycl::exceptions(errc) if the call was not successful.
  template <sycl::errc errc, OlApiKind OlApiOffset, typename... ArgsT>
  void call(ArgsT &&...Args) const {
    auto Err = call_nocheck<OlApiOffset>(std::forward<ArgsT>(Args)...);
    checkOlResult<errc>(Err);
  }

private:
  OlFuncPtrMapT OlFuncPtrs;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
