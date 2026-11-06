// REQUIRES: target-spir
// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with atomic16 support
// RUN: %{build} %{full_atomic16_flags} -o %t.out
// RUN: %if aspect-ext_oneapi_atomic16 && aspect-fp16 %{ %{run} %t.out %} %else %{ echo "Skipped: requires atomic16 and fp16" %}

#include "add.h"

int main() { add_test_all<access::address_space::generic_space>(); }
