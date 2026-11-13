// RUN: %{build} -Wno-error=deprecated-declarations -o %t.out -fno-sycl-id-queries-fit-in-int
// RUN: %{run} %t.out
//
// REQUIRES: cuda

#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/max_work_groups.hpp>

using namespace sycl;

const size_t lsize = 32;
const std::string expected_msg =
    "Number of work-groups exceed limit for dimension ";

template <int N>
void check(range<N> global, range<N> local, bool expect_fail = false) {
  queue q{[](exception_list Errors) {
    for (auto Error : Errors)
      std::rethrow_exception(Error);
  }};
  bool caught = false;
  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for(nd_range<N>(global, local), [=](nd_item<N> item) {});
    });
    q.wait_and_throw();
  } catch (const exception &e) {
    caught = true;
    if (expect_fail) {
      assert(e.code() == errc::nd_range);
      std::string msg = e.what();
      assert(msg.rfind(expected_msg, 0) == 0);
    } else {
      throw;
    }
  }
  assert(caught == expect_fail && "Expected an over-limit launch to fail");
}

int main() {
  queue q;
  device d = q.get_device();
  id<1> max_1 = d.get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<1>>();
  // Boundary acceptance is covered with mocked limits in WorkGroupLimits.cpp;
  // real devices only execute one work-group per positive case.
  check(range<1>(lsize), range<1>(lsize));
  check(range<1>((max_1[0] + 1) * lsize), range<1>(lsize), true);

  id<2> max_2 = d.get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<2>>();
  check(range<2>(1, lsize), range<2>(1, lsize));
  check(range<2>(1, (max_2[1] + 1) * lsize), range<2>(1, lsize), true);
  check(range<2>(lsize, 1), range<2>(lsize, 1));
  check(range<2>((max_2[0] + 1) * lsize, 1), range<2>(lsize, 1), true);

  id<3> max_3 = d.get_info<
      sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
  check(range<3>(1, 1, lsize), range<3>(1, 1, lsize));
  check(range<3>(1, 1, (max_3[2] + 1) * lsize), range<3>(1, 1, lsize), true);
  check(range<3>(1, lsize, 1), range<3>(1, lsize, 1));
  check(range<3>(1, (max_3[1] + 1) * lsize, 1), range<3>(1, lsize, 1), true);
  check(range<3>(lsize, 1, 1), range<3>(lsize, 1, 1));
  check(range<3>((max_3[0] + 1) * lsize, 1, 1), range<3>(lsize, 1, 1), true);
}
