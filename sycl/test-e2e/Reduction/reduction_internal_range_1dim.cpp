// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "reduction_internal.hpp"

int main() {
  queue q;
  RedStorage Storage(q);

  testRange(Storage, range<1>{42});

  return 0;
}
// REQUIRES: aspect-usm_device_allocations
