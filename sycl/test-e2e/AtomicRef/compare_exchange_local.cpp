// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build} -o %t.out
// RUN: %{build} %{full_atomic_flags} -o %t.full.out
// RUN: %if aspect-ext_oneapi_atomic16 && aspect-atomic64 && aspect-fp64 %{ %{run} %t.full.out %} %else %{ %{run} %t.out %}

#include "compare_exchange.h"

int main() { compare_exchange_test_all<access::address_space::local_space>(); }
