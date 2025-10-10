#include <sycl/sycl.hpp>

int main() {
    auto plts = sycl::platform::get_platforms();
    std::cout << " Found " << plts.size() << " platforms\n";

    for (auto &P : plts) {
        // std::cout << "Platform: " << P.get_info<sycl::info::platform::name>()
        //           << " backend: " << static_cast<int>(P.get_backend())
        //           << std::endl;
        auto devs = P.get_devices();
        std::cout << " Found " << devs.size() << " devices\n";
        // for (auto &D : devs) {
        //     std::cout << "    Device: "
        //               << D.get_info<sycl::info::device::name>()
        //               << " Type: "
        //               << static_cast<int>(
        //                      D.get_info<sycl::info::device::device_type>())
        //               << std::endl;
        // }
    }
    return 0;
}