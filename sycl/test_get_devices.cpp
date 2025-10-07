#include <sycl/sycl.hpp>

int main() {
    auto plts = sycl::platform::get_platforms();
    std::cout << " Found " << plts.size() << " platforms\n";
    return 0;
}