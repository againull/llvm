// REQUIRES: aspect-ext_oneapi_queue_profiling_tag
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Tests that a profiling tag event still reports its timestamp when the
// previous tag's event was destroyed before this tag was submitted. Releasing
// the previous event lets a backend recycle its internal event object for the
// new tag while the released tag's timestamp write is still in flight; that
// write must not disturb the state of the new tag.

// HIP backend currently returns invalid values for submission time queries.
// UNSUPPORTED: hip
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/12904

#include "common.hpp"

#include <memory>
#include <vector>

constexpr size_t NumIters = 200;
constexpr size_t NumTagsPerIter = 10;

int main() {
  sycl::queue Queue{sycl::property::queue::in_order()};

  int Failures = 0;
  std::vector<sycl::event> KeptEvents;
  KeptEvents.reserve(NumTagsPerIter);

  for (size_t I = 0; I < NumIters; ++I) {
    KeptEvents.clear();
    for (size_t J = 0; J < NumTagsPerIter; ++J) {
      // Submit a tag and drop its event straight away, without waiting for it.
      auto DroppedEvent = std::make_unique<sycl::event>(
          sycl::ext::oneapi::experimental::submit_profiling_tag(Queue));
      DroppedEvent.reset();

      KeptEvents.push_back(
          sycl::ext::oneapi::experimental::submit_profiling_tag(Queue));
    }

    Queue.wait_and_throw();

    for (sycl::event &E : KeptEvents) {
      uint64_t End =
          E.get_profiling_info<sycl::info::event_profiling::command_end>();
      CHECK(Failures, End != 0)
    }
  }

  return Failures;
}
