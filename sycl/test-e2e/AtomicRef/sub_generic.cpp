// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define TEST_ATOMIC_FLOAT
#include "sub.h"

int main() { sub_test_all<access::address_space::generic_space>(); }
