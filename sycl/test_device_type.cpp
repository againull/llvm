#include <sycl/sycl.hpp>

int main() {
    // std::cout << " All Devices: " << std::endl;
    // for (auto &P : sycl::platform::get_platforms()) {
    //     if (P.get_backend() == sycl::backend::opencl) {
    //         for (auto &D : P.get_devices()) {
    //             std::cout << "    Device: "
    //                       << D.get_info<sycl::info::device::name>()
    //                       << " Type: "
    //                       << static_cast<int>(
    //                              D.get_info<sycl::info::device::device_type>())
    //                       << std::endl;
    //         }

    //     } 
    // }
    // std::cout << " All GPU Devices: " << std::endl;
    // for (auto &P : sycl::platform::get_platforms()) {
    //     if (P.get_backend() == sycl::backend::opencl) {

    //         for (auto &D : P.get_devices(sycl::info::device_type::gpu)) {
    //             std::cout << "    Device: "
    //                       << D.get_info<sycl::info::device::name>()
    //                       << " Type: "
    //                       << static_cast<int>(
    //                              D.get_info<sycl::info::device::device_type>())
    //                       << std::endl;
    //         }
    //     } 
    // }

    std::cout << " All CPU Devices: " << std::endl;
    for (auto &P : sycl::platform::get_platforms()) {
        if (P.get_backend() == sycl::backend::opencl) {
            for (auto &D : P.get_devices(sycl::info::device_type::cpu)) {
                std::cout << "    Device: "
                          << D.get_info<sycl::info::device::name>()
                          << " Type: "  
                            << static_cast<int>(
                                     D.get_info<sycl::info::device::device_type>()) 
                            << std::endl;
            }
        } 
    }
    return 0;
}