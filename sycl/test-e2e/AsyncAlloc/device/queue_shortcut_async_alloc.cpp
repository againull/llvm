// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// Extra run to check for leaks in Level Zero using UR_L0_LEAKS_DEBUG
// RUN: %if level_zero %{%{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s --implicit-check-not=LEAK %}

// Tests async_malloc, async_malloc_from_pool and async_free when they are
// called on a queue or on a handler, in both cases without requesting an event.
// Each scenario allocates, fills the allocation from a kernel, copies it back
// and frees it.

#include <chrono>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <thread>

#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

constexpr size_t Width = 8;

// Fills the allocation with the global ids and copies it back to Out. The copy
// depends on the kernel explicitly, as an out-of-order queue does not order the
// two.
template <typename KernelName>
void fillAndCopyBack(sycl::queue &Q, void *Alloc, std::vector<char> &Out) {
  sycl::event Fill =
      Q.parallel_for<KernelName>(sycl::range<1>{Width}, [=](sycl::id<1> Id) {
        static_cast<char *>(Alloc)[Id] = static_cast<char>(Id);
      });
  Q.memcpy(Out.data(), Alloc, Width, Fill);
}

bool validate(const std::vector<char> &Out, const char *Name) {
  for (size_t I = 0; I < Width; ++I) {
    if (Out[I] != static_cast<char>(I)) {
      std::cerr << Name << ": result mismatch at " << I << "! Expected: " << I
                << ", actual: " << static_cast<int>(Out[I]) << std::endl;
      return false;
    }
  }
  return true;
}

class InOrderKernel;
class InOrderPoolKernel;
class HostTaskKernel;
class OutOfOrderKernel;
class HandlerKernel;
class InOrderHandlerHostTaskKernel;
class OutOfOrderHandlerKernel;
class OutOfOrderHandlerPoolKernel;
class OutOfOrderHandlerHostTaskKernel;

int main() {
  bool Pass = true;

  {
    // In-order queue: the queue itself orders everything.
    sycl::queue Q{sycl::property::queue::in_order{}};
    std::vector<char> Out(Width, 0);

    void *Alloc = syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);
    fillAndCopyBack<InOrderKernel>(Q, Alloc, Out);
    syclexp::async_free(Q, Alloc);
    Q.wait_and_throw();

    Pass &= validate(Out, "in-order");
  }

  {
    // Allocating from an explicit memory pool, repeatedly.
    sycl::queue Q{sycl::property::queue::in_order{}};
    syclexp::memory_pool Pool{Q.get_context(), Q.get_device(),
                              sycl::usm::alloc::device};
    std::vector<char> Out(Width, 0);

    // Allocate and free repeatedly to make sure that no state is accumulated
    // between the submissions.
    for (int I = 0; I < 4; ++I) {
      void *Alloc = syclexp::async_malloc_from_pool(Q, Width, Pool);
      fillAndCopyBack<InOrderPoolKernel>(Q, Alloc, Out);
      syclexp::async_free(Q, Alloc);
    }
    Q.wait_and_throw();

    Pass &= validate(Out, "in-order pool");
  }

  {
    // A host task cannot be ordered by the backend, so the commands after it
    // have to go through the scheduler.
    sycl::queue Q{sycl::property::queue::in_order{}};
    std::vector<char> Out(Width, 0);
    bool HostTaskExecuted = false;

    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([&]() { HostTaskExecuted = true; });
    });

    void *Alloc = syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);
    fillAndCopyBack<HostTaskKernel>(Q, Alloc, Out);
    syclexp::async_free(Q, Alloc);
    Q.wait_and_throw();

    if (!HostTaskExecuted) {
      std::cerr << "host task: not executed!" << std::endl;
      Pass = false;
    }
    Pass &= validate(Out, "host task");
  }

  {
    // Out-of-order queue: nothing is ordered implicitly, so the ordering is
    // requested with barriers.
    sycl::queue Q;
    std::vector<char> Out(Width, 0);

    void *Alloc = syclexp::async_malloc(Q, sycl::usm::alloc::device, Width);
    Q.ext_oneapi_submit_barrier();
    fillAndCopyBack<OutOfOrderKernel>(Q, Alloc, Out);
    Q.ext_oneapi_submit_barrier();
    syclexp::async_free(Q, Alloc);
    Q.wait_and_throw();

    Pass &= validate(Out, "out-of-order");
  }

  {
    // The handler overloads submitted without requesting an event must not
    // create, and thereby leak, an event either.
    sycl::queue Q{sycl::property::queue::in_order{}};
    std::vector<char> Out(Width, 0);

    for (int I = 0; I < 4; ++I) {
      void *Alloc = nullptr;
      syclexp::submit(Q, [&](sycl::handler &CGH) {
        Alloc = syclexp::async_malloc(CGH, sycl::usm::alloc::device, Width);
      });
      fillAndCopyBack<HandlerKernel>(Q, Alloc, Out);
      syclexp::submit(
          Q, [&](sycl::handler &CGH) { syclexp::async_free(CGH, Alloc); });
    }
    Q.wait_and_throw();

    Pass &= validate(Out, "handler");
  }

  {
    // The handler overload on an in-order queue after a host task: the
    // allocation is enqueued while the command group function runs, but the
    // commands submitted after it must still run after the host task.
    sycl::queue Q{sycl::property::queue::in_order{}};
    int *Value = sycl::malloc_host<int>(1, Q);
    *Value = 0;
    int Out = 0;

    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        *Value = 42;
      });
    });
    int *Alloc = nullptr;
    syclexp::submit(Q, [&](sycl::handler &CGH) {
      Alloc = static_cast<int *>(
          syclexp::async_malloc(CGH, sycl::usm::alloc::device, sizeof(int)));
    });
    syclexp::submit(Q, [&](sycl::handler &CGH) {
      CGH.single_task<InOrderHandlerHostTaskKernel>([=]() { *Alloc = *Value; });
    });
    Q.memcpy(&Out, Alloc, sizeof(int));
    syclexp::submit(
        Q, [&](sycl::handler &CGH) { syclexp::async_free(CGH, Alloc); });
    Q.wait_and_throw();
    sycl::free(Value, Q);

    if (Out != 42) {
      std::cerr << "in-order handler host task: result mismatch! Expected: 42"
                << ", actual: " << Out << std::endl;
      Pass = false;
    }
  }

  {
    // The handler overloads on an out-of-order queue, submitted without
    // requesting an event, ordered with barriers.
    sycl::queue Q;
    syclexp::memory_pool Pool{Q.get_context(), Q.get_device(),
                              sycl::usm::alloc::device};
    std::vector<char> Out(Width, 0);
    std::vector<char> PoolOut(Width, 0);

    for (int I = 0; I < 4; ++I) {
      void *Alloc = nullptr;
      void *PoolAlloc = nullptr;
      syclexp::submit(Q, [&](sycl::handler &CGH) {
        Alloc = syclexp::async_malloc(CGH, sycl::usm::alloc::device, Width);
      });
      syclexp::submit(Q, [&](sycl::handler &CGH) {
        PoolAlloc = syclexp::async_malloc_from_pool(CGH, Width, Pool);
      });
      Q.ext_oneapi_submit_barrier();
      fillAndCopyBack<OutOfOrderHandlerKernel>(Q, Alloc, Out);
      fillAndCopyBack<OutOfOrderHandlerPoolKernel>(Q, PoolAlloc, PoolOut);
      Q.ext_oneapi_submit_barrier();
      syclexp::submit(
          Q, [&](sycl::handler &CGH) { syclexp::async_free(CGH, Alloc); });
      syclexp::submit(
          Q, [&](sycl::handler &CGH) { syclexp::async_free(CGH, PoolAlloc); });
    }
    Q.wait_and_throw();

    Pass &= validate(Out, "out-of-order handler");
    Pass &= validate(PoolOut, "out-of-order handler pool");
  }

  {
    // The handler overload on an out-of-order queue, depending on a host task,
    // which makes the submission go through the scheduler.
    sycl::queue Q;
    std::vector<char> Out(Width, 0);
    bool HostTaskExecuted = false;

    sycl::event HostTask = Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([&]() { HostTaskExecuted = true; });
    });

    void *Alloc = nullptr;
    syclexp::submit(Q, [&](sycl::handler &CGH) {
      CGH.depends_on(HostTask);
      Alloc = syclexp::async_malloc(CGH, sycl::usm::alloc::device, Width);
    });
    Q.ext_oneapi_submit_barrier();
    fillAndCopyBack<OutOfOrderHandlerHostTaskKernel>(Q, Alloc, Out);
    Q.ext_oneapi_submit_barrier();
    syclexp::submit(
        Q, [&](sycl::handler &CGH) { syclexp::async_free(CGH, Alloc); });
    Q.wait_and_throw();

    if (!HostTaskExecuted) {
      std::cerr << "out-of-order handler host task: not executed!" << std::endl;
      Pass = false;
    }
    Pass &= validate(Out, "out-of-order handler host task");
  }

  if (!Pass) {
    std::cerr << "Test failed!" << std::endl;
    return 1;
  }

  std::cout << "Test passed!" << std::endl;
  return 0;
}
