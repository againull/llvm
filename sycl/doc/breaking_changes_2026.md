# SYCL Breaking Changes - 2026 Release

This document lists all user-visible breaking changes in the 2026 release of DPC++/SYCL, with migration guidance and code examples.

---

## Table of Contents

1. [The following fpga extensions have been removed from SYCL RT](#1-the-following-fpga-extensions-have-been-removed-from-sycl-rt)
2. [`sycl_ext_oneapi_annotated_arg` Extension Removal](#2-sycl_ext_oneapi_annotated_arg-extension-removal)
3. [Kernel Launch Queries Updated (experimental extension)](#3-kernel-launch-queries-updated-experimental-extension)
4. [get_backend_info() Method Removed](#4-get_backend_info-method-removed)
5. [Graph Dynamic Parameter Constructors Simplified (experimental extension)](#5-graph-dynamic-parameter-constructors-simplified-experimental-extension)
6. [Logical Operations Now Return bool](#6-logical-operations-now-return-bool)
7. [vec API changes](#7-vec-api-changes)
8. [Intel USM Address Spaces Extension Removed](#8-intel-usm-address-spaces-extension-removed)
9. [XPTI API breaking changes](#9-xpti-api-breaking-changes)
10. [XPTI Debug Stream Added](#10-xpti-debug-stream-added)
11. [Fallback Assert Implementation Removed](#11-fallback-assert-implementation-removed)
12. [SYCLCompat is discontinued](#12-syclcompat-is-discontinued)
13. [Deprecated Handler Enqueue Functions Removed](#13-deprecated-handler-enqueue-functions-removed)
14. [Printf Variadic Implementation Removed](#14-printf-variadic-implementation-removed)

---

## 1. The following fpga extensions have been removed from SYCL RT:
- `sycl_ext_intel_buffer_location`
- `sycl_ext_intel_runtime_buffer_location`
- `sycl_ext_intel_data_flow_pipes_properties`
- `sycl_ext_intel_dataflow_pipes`
- `sycl_ext_intel_fpga_datapath`
- `sycl_ext_intel_fpga_device_selector`
- `sycl_ext_intel_fpga_kernel_arg_properties`
- `sycl_ext_intel_fpga_kernel_interface_properties`
- `sycl_ext_intel_fpga_lsu`
- `sycl_ext_intel_fpga_mem`
- `sycl_ext_intel_fpga_reg`
- `sycl_ext_intel_fpga_task_sequence`
- `sycl_ext_intel_mem_channel_property`
- `sycl_ext_oneapi_annotated_arg`
- `sycl_ext_intel_usm_address_spaces`
- Removed `init_mode` and `implement_in_csr` from
    `sycl_ext_oneapi_device_global`.

## 2. `sycl_ext_oneapi_annotated_arg` Extension Removal

**Summary:** The `sycl_ext_oneapi_annotated_arg` extension has been removed.
This extension was added in order to decorate kernel arguments with properties, which were mostly specific to FPGA.
The non-fpga code which used this extension to decorate kernel arguments with the `alignment` or `unaliased` properties needs to migrate to
`sycl_ext_oneapi_annotated_ptr` extenions, it is more suitable for this purpose because these properties apply only to pointers.

### Key Differences

| Operation | annotated_arg | annotated_ptr |
|-----------|--------------|---------------|
| **Device construction** | No | Yes |
| **Implicit conversion** | Yes | No - use `.get()` |
| **operator[] with alignment** | Works | Disabled - use `.get()` |
| **operator->** | Works | Not supported - use `.get()` |


### Migration: With Alignment property

```cpp
// OLD (REMOVED)
#include <sycl/ext/oneapi/experimental/annotated_arg.hpp>
auto arg = annotated_arg(data, alignment<128>);
q.parallel_for(range<1>{N}, [=](id<1> idx) {
    arg[idx] = value;
});

// NEW (USE THIS)
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
q.parallel_for(range<1>{N}, [=](id<1> idx) {
    auto ptr = annotated_ptr(data, alignment<128>);
    ptr.get()[idx] = value;  // Use .get() with alignment property
});
```

### Migration: Without Alignment Property

```cpp
q.parallel_for(range<1>{N}, [=](id<1> idx) {
    auto in_ptr = annotated_ptr(in, unaliased);   // No alignment property
    auto out_ptr = annotated_ptr(out, unaliased);

    out_ptr[idx] = in_ptr[idx] + 100;  // operator[] works without alignment property
});
```

### Migration: Struct Member Access

```cpp
struct Point { int x, y, z; };

q.parallel_for(range<1>{N}, [=](id<1> idx) {
    auto ptr = annotated_ptr(points, alignment<64>);

    // OLD: ptr[idx]->x = i;  // operator-> not supported

    // NEW: Use .get()
    ptr.get()[idx].x = i;
    ptr.get()[idx].y = i * 2;
    ptr.get()[idx].z = i * 3;
});
```

**Recommendation:** Always use `.get()` for consistent behavior across all scenarios.

---

## 3. Kernel Launch Queries Updated (experimental extension)

**Summary:** The deprecated `max_num_work_group_sync` query has been removed. It is replaced with `max_num_work_groups` which takes extra parameters for local work-group size and dynamic local memory size (in bytes), so that those runtime resource limiting factors are taken into account in the final group count suggestion.

### Removed Query

```cpp
namespace syclex = sycl::ext::oneapi::experimental;

// OLD (REMOVED)
size_t maxGroups = kernel.ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_num_work_group_sync>(queue);
```

### Migration: Use max_num_work_groups

```cpp
namespace syclex = sycl::ext::oneapi::experimental;

// NEW (USE THIS)
// Specify your work group configuration
sycl::range<3> workGroupSize{256, 1, 1}; // work group size
size_t dynamicLocalMemorySize = 0;  // amount of dynamic work-group local memory (in bytes), accounting for any kernel properties or features.

size_t maxConcurrentWorkGroups = kernel.ext_oneapi_get_info<
    syclex::info::kernel_queue_specific::max_num_work_groups>(
        queue, workGroupSize, dynamicLocalMemorySize);

// Returns 0 if configuration exceeds hardware limits
```

---

## 4. get_backend_info() Method Removed

**Summary:** The deprecated `get_backend_info()` method has been removed from all SYCL objects because they are returning info descriptors that are not backend-specific but SYCL core info descriptors. The standard `get_info()` method needs to be used instead.

### Migration

```cpp
// OLD (REMOVED)
sycl::device dev;
std::string version = dev.get_backend_info<sycl::info::device::version>();
std::string backend_ver = dev.get_backend_info<sycl::info::device::backend_version>();

sycl::platform plat;
std::string plat_version = plat.get_backend_info<sycl::info::platform::version>();

// NEW (USE THIS)
sycl::device dev;
std::string version = dev.get_info<sycl::info::device::version>();
std::string backend_ver = dev.get_info<sycl::info::device::backend_version>();

sycl::platform plat;
std::string plat_version = plat.get_info<sycl::info::platform::version>();
```

### Affected Objects

All SYCL objects: `platform`, `device`, `context`, `queue`, `event`, `kernel`.

---

## 5. Graph Dynamic Parameter Constructors Simplified (experimental extension)

**Summary:** The `command_graph` parameter has been removed from dynamic parameter constructors. Graph association is now automatic.

### Migration: Dynamic Parameters

```cpp
namespace exp_ext = sycl::ext::oneapi::experimental;

// OLD (REMOVED)
exp_ext::command_graph<exp_ext::graph_state::modifiable> graph;
int* ptr = sycl::malloc_device<int>(100, queue);
exp_ext::dynamic_parameter inputParam(graph, ptr);

// NEW (USE THIS)
int* ptr = sycl::malloc_device<int>(100, queue);
exp_ext::dynamic_parameter inputParam(ptr);  // Graph parameter removed
```

### Migration: Dynamic Work Group Memory

```cpp
// OLD (REMOVED)
size_t localSize = 256;
exp_ext::dynamic_work_group_memory<int[]> dynMem(graph, localSize);

// NEW (USE THIS)
exp_ext::dynamic_work_group_memory<int[]> dynMem(localSize);
```

### Migration: Dynamic Local Accessor

```cpp
// OLD (REMOVED)
sycl::range<1> allocationSize{256};
exp_ext::dynamic_local_accessor<int, 1> dynAcc(graph, allocationSize);

// NEW (USE THIS)
exp_ext::dynamic_local_accessor<int, 1> dynAcc(allocationSize);
```

---

## 6. Logical Operations Now Return bool

**Summary:** `sycl::logical_and<T>` and `sycl::logical_or<T>` now always return `bool`, matching C++ standard library behavior.

### Migration: Variable Types

```cpp
// OLD (COMPILED, BUT DEPRECATED)
int result = sycl::logical_and<int>{}(a, b);

// NEW (USE THIS)
bool result = sycl::logical_and<int>{}(a, b);
```

### Migration: Group Algorithms

```cpp
// OLD (NO LONGER COMPILES)
int result = sycl::reduce_over_group(
    group, 0, sycl::logical_and<int>{});

// NEW (USE THIS)
bool result = sycl::reduce_over_group(
    group, true, sycl::logical_and<bool>{});
```

### Migration: Explicit Conversion

```cpp
// If you need an integer result for compatibility
int result = static_cast<int>(sycl::logical_and<int>{}(a, b));
```

---

## 7. vec API changes.

**Summary:** The `sycl::vec` API has been updated with several breaking changes:
- Implicit cross-type conversions removed (use `.convert<T>()`)
- The `vector_t` type has been removed

## Benefits

- **Specification Compliance**: Aligns with SYCL 2020
- **Portability**: Consistent across implementations
- **Type Safety**: Prevents silent precision loss
- **Clarity**: Explicit conversions show intent

### Migration: Basic Conversion

```cpp
// OLD (NO LONGER COMPILES)
vec<half, 1> half_vec{2.5f};
vec<float, 1> float_vec;
float_vec = half_vec;  // Implicit conversion

// NEW (USE THIS)
vec<half, 1> half_vec{2.5f};
auto float_vec = half_vec.convert<float>();  // Explicit conversion
```

### Migration: Multi-Element Vectors

```cpp
// OLD (NO LONGER COMPILES)
vec<half, 4> half_data{1.0f, 2.0f, 3.0f, 4.0f};
vec<float, 4> float_data = half_data;

// NEW (USE THIS)
vec<half, 4> half_data{1.0f, 2.0f, 3.0f, 4.0f};
auto float_data = half_data.convert<float>();
```

### Migration: Function Parameters

```cpp
void process(vec<float, 4> data);

vec<half, 4> input{1, 2, 3, 4};

// OLD (NO LONGER COMPILES)
process(input);  // Implicit conversion

// NEW (USE THIS)
process(input.convert<float>());
```

### Migration: Swizzle with Type Change

```cpp
vec<half, 4> rgba{1.0f, 2.0f, 3.0f, 4.0f};

// OLD (NO LONGER COMPILES)
vec<float, 1> blue = rgba.swizzle<2>();

// NEW (USE THIS)
vec<float, 1> blue{static_cast<float>(rgba[2])};

// OR convert first if doing multiple operations
auto rgba_float = rgba.convert<float>();
auto blue2 = rgba_float.swizzle<2>();
```

### Common Conversion Types

```cpp
vec<half, N> hv;
vec<int, N> iv;
vec<float, N> fv;
vec<double, N> dv;

auto f_from_h = hv.convert<float>();   // half → float
auto h_from_f = fv.convert<half>();    // float → half
auto f_from_i = iv.convert<float>();   // int → float
auto i_from_f = fv.convert<int>();     // float → int (truncates)
auto f_from_d = dv.convert<float>();   // double → float
auto d_from_f = fv.convert<double>();  // float → double
```

### Removed: vector_t Type

```cpp
// OLD (REMOVED)
vec<float, 4> v{1, 2, 3, 4};
using VecType = vec<float, 4>::vector_t;
VecType native = v.get_vector_t();

// NEW (USE THIS)
vec<float, 4> v{1, 2, 3, 4};
// Just use sycl::vec directly - vector_t is removed
```

---

## 8. Intel USM Address Spaces Extension Removed

**Summary:** The `sycl_ext_intel_usm_address_spaces` extension has been removed. Use standard SYCL `global_space` address space instead.

### What Was Removed

- Address spaces: `ext_intel_global_device_space`, `ext_intel_global_host_space`
- Type aliases: `ext::intel::device_ptr`, `ext::intel::host_ptr`, `ext::intel::raw_device_ptr`, etc.
- Header: `sycl/ext/intel/usm_pointers.hpp`

### Why This Change?

- The separate device/host address spaces were designed for FPGA use cases but provide no advantage on GPUs or CPUs.
- Standard SYCL is sufficient: The standard global_space address space works for all USM allocations (device, host, and shared).

### Migration Table

| Old Extension Type | New Standard Type |
|--------------------|-------------------|
| `ext::intel::device_ptr<T>` | `sycl::global_ptr<T>` |
| `ext::intel::host_ptr<T>` | `sycl::global_ptr<T>` |
| `ext::intel::raw_device_ptr<T>` | `sycl::multi_ptr<T, global_space, decorated::no>` |
| `ext::intel::raw_host_ptr<T>` | `sycl::multi_ptr<T, global_space, decorated::no>` |
| `ext::intel::decorated_device_ptr<T>` | `sycl::multi_ptr<T, global_space, decorated::yes>` |
| `ext::intel::decorated_host_ptr<T>` | `sycl::multi_ptr<T, global_space, decorated::yes>` |

### Migration: Basic Usage

```cpp
// OLD (REMOVED)
#include <sycl/ext/intel/usm_pointers.hpp>
sycl::ext::intel::device_ptr<int> ptr;

// NEW (USE THIS)
sycl::global_ptr<int> ptr;
```

### Migration: Address Space Specifications

```cpp
// OLD (REMOVED)
sycl::multi_ptr<int, sycl::access::address_space::ext_intel_global_device_space>

// NEW (USE THIS)
sycl::multi_ptr<int, sycl::access::address_space::global_space>
```

### Best Practice: USM with Raw Pointers

```cpp
// USM pointers work directly
int *usm_ptr = sycl::malloc_device<int>(1024, q);
q.parallel_for(range, [=](id<1> idx) {
    usm_ptr[idx] = idx[0];  // Direct pointer usage
});
```

### Best Practice: Use Accessors Directly

```cpp
q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for(range, [=](id<1> idx) {
        acc[idx] = idx[0];  // Direct accessor usage
    });
});
```

---

## 9. XPTI API breaking changes

**Summary:** String ID type changed from `int32_t` to `uint32_t`.

### Breaking Change: String ID Type

```cpp
const xpti::payload_t *payload = /* get payload */;

// OLD (NO LONGER COMPILES)
int32_t name_id = payload->name_sid();
int32_t file_id = payload->source_file_sid();

// NEW (USE THIS)
uint32_t name_id = payload->name_sid();
uint32_t file_id = payload->source_file_sid();
```


## 10. XPTI Debug Stream Added

**Summary:** A new XPTI stream `"sycl.debug"` has been introduced to separate debug information from performance-critical tracing.

### For Tool Developers: Stream Selection

```cpp
XPTI_CALLBACK_API void xptiTraceInit(unsigned int major, unsigned int minor,
                                      const char *version, const char *stream) {
    // For low-overhead performance profiling (15-20% faster)
    if (std::string_view(stream) == "sycl") {
        uint8_t streamID = xptiRegisterStream(stream);
        // Register callbacks - minimal metadata
    }

    // For detailed debugging (comprehensive metadata)
    if (std::string_view(stream) == "sycl.debug") {
        uint8_t streamID = xptiRegisterStream(stream);
        // Register callbacks - full metadata
    }

    // Can handle both streams
    if (std::string_view(stream) == "sycl" ||
        std::string_view(stream) == "sycl.debug") {
        // Handle both streams
    }
}
```

### Stream Selection Guide

**Performance-oriented tools (recommended):**
- Subscribe to `"sycl"` stream only
- Essential metadata only

**Debug-oriented tools:**
- Subscribe to `"sycl.debug"` stream
- Comprehensive metadata
- Higher overhead acceptable

**Note:** When subscribing to both, only notifications from `"sycl.debug"` are delivered to avoid duplication.

### Similar Changes for Unified Runtime

New stream: `"ur.call.debug"` (alongside existing `"ur.call"`)

```cpp
if (std::string_view(stream) == "ur.call" ||
    std::string_view(stream) == "ur.call.debug") {
    // Handle UR API tracing
}
```

### User Impact

**For SYCL application developers:** No changes required.

**For profiler/tracer tool developers:** Update tools to subscribe to appropriate stream based on use case.

---

## 11. Fallback Assert Implementation Removed

**Summary:** The fallback assertion implementation has been removed. Devices without native assert support now **silently ignore** assertions instead of checking them after kernel completion.

### What Changed

Previously, SYCL provided two implementations for `assert()` in device code:

1. **Native asserts** (preferred): Backend provides direct assertion support
2. **Fallback asserts** (deprecated, now removed): Runtime checked assertions after kernel completion using auxiliary kernels

The fallback implementation has been removed. Now:
- Devices **with** native assert support: Assertions work immediately and reliably
- Devices **without** native assert support: Assertions are **silently ignored**

### Checking Device Support

Query the `aspect::ext_oneapi_native_assert` aspect to check if assertions are supported:

```cpp
#include <sycl/sycl.hpp>

sycl::queue q;
sycl::device dev = q.get_device();

if (dev.has(sycl::aspect::ext_oneapi_native_assert)) {
    std::cout << "Device supports native asserts - assert() will work\n";
} else {
    std::cout << "Device does NOT support native asserts - assert() will be IGNORED\n";
}
```

### User Impact

**For most users:** The `assert()` API itself is unchanged:

```cpp
#include <cassert>
#include <sycl/sycl.hpp>

sycl::queue q;
q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
        // This works exactly as before on devices with native assert support
        assert(idx[0] < 1024 && "Index out of bounds");

        // On devices WITHOUT native assert support, this is now IGNORED
        // (previously, it would have been checked after kernel completion)
    });
});
```

### Migration Guide

**If you rely on assertions for correctness:**

```cpp
#include <cassert>
#include <sycl/sycl.hpp>

sycl::queue q;

// Check if asserts are supported
if (!q.get_device().has(sycl::aspect::ext_oneapi_native_assert)) {
    std::cerr << "Warning: Device does not support native asserts.\n";
    std::cerr << "Assertions in kernels will be ignored.\n";

    // Options:
    // 1. Use a different device that supports native asserts
    // 2. Implement alternative validation (e.g., return error codes)
    // 3. Run with assertions disabled (define NDEBUG)
}

q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
        assert(idx[0] < 1024);  // Only works on devices with native support
    });
});
```

**Alternative: Manual error checking for devices without native assert support:**

```cpp
// Instead of relying on assert(), implement explicit error handling
q.submit([&](sycl::handler &cgh) {
    auto error_flag = sycl::malloc_shared<int>(1, q);
    *error_flag = 0;

    cgh.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
        if (idx[0] >= 1024) {
            *error_flag = 1;  // Signal error
            return;
        }
        // ... continue processing
    });
}).wait();

if (*error_flag) {
    std::cerr << "Error detected in kernel\n";
    std::abort();
}
sycl::free(error_flag, q);
```

### Why This Change?

1. **Performance:** Fallback asserts added overhead with auxiliary kernels and host tasks
2. **Maintenance:** Simplified implementation by removing complex fallback infrastructure

---

## 12. SYCLCompat is discontinued.
See presentation.

---

## 13. Deprecated Handler Enqueue Functions Removed

**Summary:** The deprecated `handler` enqueue functions that take both a precompiled `kernel` object and a lambda have been removed. These overloads were not part of SYCL 2020 specification.

### What Was Removed

The following deprecated function overloads have been removed from `sycl::handler`:

```cpp
// REMOVED: single_task with kernel + lambda
template <typename KernelName = detail::auto_name, typename KernelType>
void single_task(kernel Kernel, const KernelType &KernelFunc);

// REMOVED: parallel_for with kernel + lambda
template <typename KernelName = detail::auto_name, typename KernelType, int Dims>
void parallel_for(kernel Kernel, range<Dims> NumWorkItems,
                  const KernelType &KernelFunc);

// REMOVED: parallel_for with kernel + lambda + offset
template <typename KernelName = detail::auto_name, typename KernelType, int Dims>
void parallel_for(kernel Kernel, range<Dims> NumWorkItems,
                  id<Dims> WorkItemOffset, const KernelType &KernelFunc);

// REMOVED: parallel_for with kernel + lambda + nd_range
template <typename KernelName = detail::auto_name, typename KernelType, int Dims>
void parallel_for(kernel Kernel, nd_range<Dims> NDRange,
                  const KernelType &KernelFunc);

// REMOVED: parallel_for_work_group with kernel + lambda
template <typename KernelName = detail::auto_name, typename KernelType, int Dims>
void parallel_for_work_group(kernel Kernel, range<Dims> NumWorkGroups,
                             const KernelType &KernelFunc);

// REMOVED: parallel_for_work_group with kernel + lambda + work group size
template <typename KernelName = detail::auto_name, typename KernelType, int Dims>
void parallel_for_work_group(kernel Kernel, range<Dims> NumWorkGroups,
                             range<Dims> WorkGroupSize,
                             const KernelType &KernelFunc);
```

### Why This Change?

These overloads were:
- **Not part of SYCL 2020 specification**
- **Deprecated** for several releases
- **Confusing:** Mixed precompiled kernels with JIT-compiled lambdas

### Migration: Use Standard Lambda-Only Overloads

```cpp
#include <sycl/sycl.hpp>

sycl::queue q;

// OLD (REMOVED) - kernel object + lambda
sycl::kernel precompiled_kernel = /* ... */;
q.submit([&](sycl::handler &cgh) {
    cgh.single_task(precompiled_kernel, [=]() {
        // kernel code
    });
});

// NEW (USE THIS) - just use the lambda
q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
        // kernel code
    });
});
```

### Migration: parallel_for Examples

```cpp
// OLD (REMOVED)
sycl::kernel precompiled_kernel = /* ... */;
q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(precompiled_kernel, sycl::range<1>{1024}, [=](sycl::id<1> idx) {
        // kernel code
    });
});

// NEW (USE THIS)
q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
        // kernel code
    });
});
```

### Migration: nd_range Example

```cpp
// OLD (REMOVED)
sycl::kernel precompiled_kernel = /* ... */;
q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(precompiled_kernel,
                     sycl::nd_range<1>{1024, 32},
                     [=](sycl::nd_item<1> item) {
        // kernel code
    });
});

// NEW (USE THIS)
q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>{1024, 32},
                     [=](sycl::nd_item<1> item) {
        // kernel code
    });
});
```

### Note on Kernel Bundles

If you need to explicitly control which kernels are used, use **kernel bundles**, for example:

#### Option 1: Use kernel bundle with lambda

```cpp
#include <sycl/sycl.hpp>

class MyKernel;

sycl::queue q;
auto ctx = q.get_context();

// Get or build kernel bundle
auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);

// Use the bundle with standard lambda overloads
q.submit([&](sycl::handler &cgh) {
    cgh.use_kernel_bundle(bundle);
    cgh.parallel_for<MyKernel>(sycl::range<1>{1024}, [=](sycl::id<1> idx) {
        // kernel code
    });
});
```

#### Option 2: Get kernel object from bundle and use with parallel_for

```cpp
#include <sycl/sycl.hpp>

class MyKernel;

sycl::queue q;
auto ctx = q.get_context();

// Submit kernel first to ensure it's compiled
q.submit([&](sycl::handler &cgh) {
    cgh.single_task<MyKernel>([=]() {});
});

// Get kernel bundle
auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);

// Get kernel object from bundle
sycl::kernel_id kid = sycl::get_kernel_id<MyKernel>();
sycl::kernel kernel = bundle.get_kernel(kid);

// Alternative: get kernel directly using template
// sycl::kernel kernel = bundle.get_kernel<MyKernel>();

// Use kernel object with parallel_for (requires set_args for arguments)
int *data = sycl::malloc_shared<int>(1024, q);
q.submit([&](sycl::handler &cgh) {
    cgh.set_args(data);  // Set kernel arguments
    cgh.parallel_for(sycl::range<1>{1024}, kernel);
}).wait();
sycl::free(data, q);
```

**Note:** When using a precompiled kernel object with `parallel_for`, you must use `set_args()` or `set_arg()` to pass arguments, as the kernel object doesn't capture variables like lambdas do.

---

## 14. Printf Variadic Implementation Removed

**Summary:** The `__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__` macro has been removed. This macro previously allowed using the deprecated C-style variadic printf implementation. The variadic template implementation is now the only available option.

### Background

Previously, SYCL provided two printf implementations:

1. **C-style variadic function** (deprecated, now removed):
   - Used C-style variadic arguments: `int printf(const char* fmt, ...)`
   - **Problem:** Promoted `float` arguments to `double`, implicitly requiring fp64 support
   - Enabled with `__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__` macro

2. **Variadic template** (now the only option):
   - Uses C++ variadic templates: `template<typename... Args> int printf(const char* fmt, Args... args)`
   - **Advantage:** Preserves exact argument types without promotion
   - Works on devices without fp64 support

The C-style variadic implementation has been completely removed in this release.

### User Impact

**For most users:** No changes needed. The `printf` API works identically:

```cpp
#include <sycl/sycl.hpp>

sycl::queue q;
q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
        // This works exactly as before - no changes needed
        sycl::ext::oneapi::experimental::printf("Hello, World! %d\n", 42);
        sycl::ext::oneapi::experimental::printf("Float: %f\n", 3.14f);
    });
});
```

**If you were using `__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__`:**

```bash
# OLD (this flag no longer has any effect)
clang++ -fsycl -D__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ mycode.cpp

# NEW (just remove the flag)
clang++ -fsycl mycode.cpp
```

Both commands now use the variadic template implementation internally.

---



## Additional Resources

- **SYCL Extensions Documentation:** `sycl/doc/extensions/`
- **Test Examples:** `sycl/test-e2e/`
- **XPTI Framework:** `xpti/doc/`

For questions or issues, refer to the intel/llvm repository documentation.



