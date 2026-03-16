# Misc

### New Feature: Explicit Conversion for Single-Element vec

**Summary:** Single-element `vec` objects can now be explicitly converted to other types using `static_cast`.

```cpp
vec<half, 1> h1{2.5f};

// NEW: Explicit conversion now supported
int i = static_cast<int>(h1);  // Works: converts to 2
float f = static_cast<float>(h1);  // Works: converts to 2.5f

// Can also use in boolean contexts
if (h1) {
    // Executes if h1[0] is non-zero
}

// Still need .convert<T>() for multi-element vectors
vec<half, 4> h4{1.0f, 2.0f, 3.0f, 4.0f};
auto f4 = h4.convert<float>();  // Required for N > 1
```

**Recommendation:** For single-element vectors, both `static_cast<T>()` and `.convert<T>()` work, but `static_cast` is more idiomatic for scalar-like conversions.


### New Feature: std::byte as vec Element Type

**Summary:** `std::byte` is now a legal element type for `sycl::vec`, with appropriate operator restrictions.

```cpp
#include <cstddef>  // For std::byte

// NOW SUPPORTED
vec<std::byte, 4> bytes{std::byte{0x01}, std::byte{0x02},
                        std::byte{0x03}, std::byte{0x04}};

// Supported operations on vec<std::byte, N>:
auto b1 = bytes & vec<std::byte, 4>{std::byte{0x0F}};  // Bitwise AND
auto b2 = bytes | vec<std::byte, 4>{std::byte{0xF0}};  // Bitwise OR
auto b3 = bytes ^ vec<std::byte, 4>{std::byte{0xFF}};  // Bitwise XOR
auto b4 = ~bytes;                                       // Bitwise NOT

// Comparison operations
bool eq = (bytes == vec<std::byte, 4>{std::byte{0x01}});  // Equality
bool ne = (bytes != vec<std::byte, 4>{std::byte{0x00}});  // Inequality
```

**Restrictions:** Arithmetic operations (`+`, `-`, `*`, `/`, `%`) are **not available** for `vec<std::byte>`, matching the behavior of plain `std::byte` in C++.

**Migration from deprecated sycl::byte:**
```cpp
// OLD (DEPRECATED in SYCL 2020)
vec<sycl::byte, 4> old_bytes;

// NEW (USE THIS)
vec<std::byte, 4> new_bytes;
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

### Named Swizzle Return Types (Specification Clarification/Bugfix)

**Summary:** The SYCL 2020 specification now clarifies that single-element named swizzles (`.x()`, `.y()`, `.z()`, `.w()`) return **references to the element** (`T&`) instead of 1-element swizzle objects.

**Impact:** This is primarily a specification clarification. Most implementations (including DPC++) already returned references, so existing code typically works unchanged.

#### What Changed

| Operation | Old Spec | New Spec | DPC++ Implementation |
|-----------|----------|----------|---------------------|
| `vec<T,4>.x()` | 1-element swizzle | `T&` | Already returns `T&` |
| `vec<T,4>.y()` | 1-element swizzle | `T&` | Already returns `T&` |
| `vec<T,4>.z()` | 1-element swizzle | `T&` | Already returns `T&` |
| `vec<T,4>.w()` | 1-element swizzle | `T&` | Already returns `T&` |

**Note:** Multi-element swizzles (e.g., `.xy()`, `.xyz()`) still return swizzle objects as before.

#### Code That Works Unchanged

```cpp
vec<uint8_t, 4> v4{255, 128, 64, 32};

// These work the same in both old and new spec
int i = v4.x() + 1;      // Result: 256 (promotes to int)
v4.y() = 200;            // Assignment works
auto val = v4.x();       // Stores uint8_t value
```

## register_alloc_mode Property Replaced (Undocumented feature in `sycl::detail` namespace)

**Summary:** The `register_alloc_mode` kernel property (`sycl::detail` namespace) has been removed. Use Intel's `grf_size` properties instead.

### Migration: Automatic Allocation

```cpp
// OLD (REMOVED)
#include <sycl/detail/kernel_properties.hpp>
properties prop{register_alloc_mode<register_alloc_mode_enum::automatic>};

// NEW (USE THIS)
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
properties prop{sycl::ext::intel::experimental::grf_size_automatic};
```

### Migration: Large Register Allocation (256-byte GRF)

```cpp
// OLD (REMOVED)
properties prop{register_alloc_mode<register_alloc_mode_enum::large>};

// NEW (USE THIS)
properties prop{sycl::ext::intel::experimental::grf_size<256>};
```

### Complete Example

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace sycl;
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;

queue q;
buffer<float> buf(1024);

// Automatic GRF size
syclex::properties prop_auto{intelex::grf_size_automatic};
q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class Kernel1>(range<1>{1024}, prop_auto,
                                     [=](id<1> idx) {
        acc[idx] = idx[0];
    });
});

// Explicit 256-byte GRF
syclex::properties prop_256{intelex::grf_size<256>};
q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class Kernel2>(range<1>{1024}, prop_256,
                                     [=](id<1> idx) {
        acc[idx] = idx[0] * 2;
    });
});
```

**Available sizes:** `grf_size_automatic`, `grf_size<128>`, `grf_size<256>`

---


## XPTI API enhancements (new features)


**Summary:** XPTI tracing APIs have been enhanced with optional code pointer tracking and improved performance.
### New Feature: Code Pointer Tracking (Optional)

```cpp
#include <xpti/xpti_trace_framework.h>

// OLD - without code pointer
xpti_tracepoint_t *tp = xptiCreateTracepoint(
    "myFunction", "myfile.cpp", 42, 10);

// NEW - with optional code pointer for better debugging
void *code_addr = __builtin_return_address(0);
xpti_tracepoint_t *tp = xptiCreateTracepoint(
    "myFunction", "myfile.cpp", 42, 10, code_addr);

// Still backward compatible - nullptr is default
xpti_tracepoint_t *tp2 = xptiCreateTracepoint(
    "myFunction", "myfile.cpp", 42, 10);
```

### Recommended: Use Payload Validation Utility

```cpp
const xpti::payload_t *payload = /* get payload */;

// OLD - manual checking
if (payload->name_sid() != xpti::invalid_id) {
    std::cout << "Name: " << payload->name << "\n";
}

// NEW - use validation utility (RECOMMENDED)
if (xpti::is_valid_payload(payload) && payload->name) {
    std::cout << "Name: " << payload->name << "\n";
}
```