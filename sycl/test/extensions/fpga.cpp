// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include <type_traits>

namespace intelfpga {
template <typename T, sycl::access::address_space space,
          sycl::access::decorated is_decorated>
void lsu_body(sycl::multi_ptr<const T, space, is_decorated> input_ptr,
              sycl::multi_ptr<T, space, is_decorated> output_ptr) {
  using PrefetchingLSU =
      sycl::ext::intel::lsu<sycl::ext::intel::prefetch<true>,
                            sycl::ext::intel::statically_coalesce<false>>;

  using BurstCoalescedLSU =
      sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>,
                            sycl::ext::intel::statically_coalesce<false>>;

  using CachingLSU =
      sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>,
                            sycl::ext::intel::cache<1024>,
                            sycl::ext::intel::statically_coalesce<false>>;

  using PipelinedLSU = sycl::ext::intel::lsu<>;

  int X = PrefetchingLSU::load(input_ptr); // int X = input_ptr[0]
  int Y = CachingLSU::load(input_ptr + 1); // int Y = input_ptr[1]

  BurstCoalescedLSU::store(output_ptr, X); // output_ptr[0] = X
  PipelinedLSU::store(output_ptr + 1, Y);  // output_ptr[1] = Y
}
} // namespace intelfpga

int main() {
  sycl::queue Queue;
  /* Check buffer_location property  */
  sycl::buffer<int, 1> Buf{sycl::range{1}};
  Queue.submit([&](sycl::handler &CGH) {
    sycl::ext::oneapi::accessor_property_list PL{
        sycl::ext::intel::buffer_location<1>};
    sycl::accessor Acc(Buf, CGH, sycl::write_only, PL);
    CGH.single_task<class Test>([=]() { Acc[0] = 42; });
  });
  Queue.wait();

  auto Acc = Buf.get_host_access();
  assert(Acc[0] == 42 && "Value mismatch");

  /*Check LSU interface*/
  {

    {
      auto *out_ptr = sycl::malloc_host<int>(1, Queue.get_context());
      auto *in_ptr = sycl::malloc_host<int>(1, Queue.get_context());
      Queue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class HostAnnotation>([=]() {
          sycl::ext::intel::host_ptr<const int> input_ptr(in_ptr);
          sycl::ext::intel::host_ptr<int> output_ptr(out_ptr);
          intelfpga::lsu_body<
              int, sycl::access::address_space::ext_intel_global_host_space>(
              input_ptr, output_ptr);
        });
      });
    }
    {
      auto *out_ptr = sycl::malloc_device<int>(1, Queue);
      auto *in_ptr = sycl::malloc_device<int>(1, Queue);
      Queue.submit([&](sycl::handler &cgh) {
        cgh.single_task<class DeviceAnnotation>([=]() {
          sycl::ext::intel::device_ptr<const int> input_ptr(in_ptr);
          sycl::ext::intel::device_ptr<int> output_ptr(out_ptr);
          intelfpga::lsu_body<
              int, sycl::access::address_space::ext_intel_global_device_space>(
              input_ptr, output_ptr);
        });
      });
    }
    {
      sycl::buffer<int, 1> output_buffer(1);
      sycl::buffer<int, 1> input_buffer(1);
      Queue.submit([&](sycl::handler &cgh) {
        auto output_accessor =
            output_buffer.get_access<sycl::access::mode::write>(cgh);
        auto input_accessor =
            input_buffer.get_access<sycl::access::mode::read>(cgh);
        cgh.single_task<class AccessorAnnotation>([=]() {
          auto input_ptr = sycl::global_ptr<const int>(input_accessor);
          auto output_ptr = sycl::global_ptr<int>(output_accessor);
          intelfpga::lsu_body<>(input_ptr, output_ptr);
        });
      });
    }
  }

  /*Check DSP control interface*/
  sycl::buffer<int, 1> output_buffer(1);
  sycl::buffer<int, 1> input_buffer(1);
  Queue.submit([&](sycl::handler &cgh) {
    auto output_accessor =
        output_buffer.get_access<sycl::access::mode::write>(cgh);
    auto input_accessor =
        input_buffer.get_access<sycl::access::mode::read>(cgh);
    cgh.single_task<class DSPControlKernel>([=]() {
      float sum = input_accessor[0];
      sycl::ext::intel::math_dsp_control<
          sycl::ext::intel::Preference::Softlogic>([&] { sum += 1.23f; });
      sycl::ext::intel::math_dsp_control<sycl::ext::intel::Preference::DSP>(
          [&] { sum += 1.23f; });
      sycl::ext::intel::math_dsp_control<
          sycl::ext::intel::Preference::Softlogic,
          sycl::ext::intel::Propagate::Off>([&] { sum += 4.56f; });
      sycl::ext::intel::math_dsp_control<sycl::ext::intel::Preference::DSP,
                                         sycl::ext::intel::Propagate::Off>(
          [&] { sum += 4.56f; });
      output_accessor[0] = sum;
    });
  });

  return 0;
}
