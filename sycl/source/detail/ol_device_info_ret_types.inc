// # curl https://raw.githubusercontent.com/oneapi-src/unified-runtime/refs/heads/main/scripts/core/device.yml 2>/dev/null \
// #  | awk '/name: .x_device_info_t/{flag=1} /^---/{flag=0} flag'  \
// #  | python a.py > ur_device_info_ret_types.inc
//
// import yaml
// import re
// import sys
//
// def get_type(x):
//     tags = x["desc"].split(" ")[0]
//     type = re.sub(r'\]\[.*', "]", tags)[1:-1]
//     type = type.replace("char[]", "std::string")
//     type = re.sub(r'(.*)\[\]', r"std::vector<\1>", type)
//     type = re.sub(r'(.*)\[(\d)\]', r"std::array<\1, \2>", type)
//     return type.replace("$x", "ur")
//
// for x in yaml.safe_load(sys.stdin)["etors"]:
//     print(f'MAP(UR_DEVICE_INFO_{x["name"]}, {get_type(x)})')
//
// With some manual post-processing

// TODO: Maybe parsing ur_api.h could be better, but YAML seems easier to
// process.

// clang-format off
MAP(OL_DEVICE_INFO_TYPE, ol_device_type_t)
MAP(OL_DEVICE_INFO_PLATFORM, ol_platform_handle_t)
MAP(OL_DEVICE_INFO_NAME, std::string)
MAP(OL_DEVICE_INFO_PRODUCT_NAME, std::string)
MAP(OL_DEVICE_INFO_VENDOR, std::string)
MAP(OL_DEVICE_INFO_DRIVER_VERSION, uint32_t)
MAP(OL_DEVICE_INFO_TYPE, ol_device_type_t)
MAP(OL_DEVICE_INFO_TYPE, ol_device_type_t)
MAP(OL_DEVICE_INFO_TYPE, ol_device_type_t)
MAP(OL_DEVICE_INFO_TYPE, ol_device_type_t)
MAP(OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE, uint32_t)
MAP(OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION, ol_dimensions_t)
MAP(OL_DEVICE_INFO_MAX_WORK_SIZE, uint32_t)
MAP(OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION, ol_dimensions_t)
MAP(OL_DEVICE_INFO_VENDOR_ID, uint32_t)
MAP(OL_DEVICE_INFO_NUM_COMPUTE_UNITS, uint32_t)
MAP(OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY, uint32_t)
MAP(OL_DEVICE_INFO_MEMORY_CLOCK_RATE, uint32_t)
MAP(OL_DEVICE_INFO_ADDRESS_BITS, uint32_t)
MAP(OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, uint64_t)
MAP(OL_DEVICE_INFO_GLOBAL_MEM_SIZE, uint64_t)
MAP(OL_DEVICE_INFO_SINGLE_FP_CONFIG, ol_device_fp_capability_flags_t)
MAP(OL_DEVICE_INFO_DOUBLE_FP_CONFIG, ol_device_fp_capability_flags_t)
MAP(OL_DEVICE_INFO_HALF_FP_CONFIG, ol_device_fp_capability_flags_t)
MAP(OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR, uint32_t)
MAP(OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT, uint32_t)
MAP(OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT, uint32_t)
MAP(OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG, uint32_t)
MAP(OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT, uint32_t)
MAP(OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE, uint32_t)
MAP(OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF, uint32_t)
// clang-format on
