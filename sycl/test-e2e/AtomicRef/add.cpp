// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define TEST_ATOMIC_FLOAT
#include "add.h"

int main() { add_test_all<access::address_space::global_space>(); }
