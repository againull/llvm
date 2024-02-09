#include <detail/kernel_bundle_impl.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
std::shared_ptr<detail::device_image_impl>
kernel_bundle_impl::select_image_for_kernel(const kernel_id &KernelID) const {
  using ImageImpl = std::shared_ptr<detail::device_image_impl>;
  // Selected image.
  ImageImpl SelectedImage = nullptr;
  // Image where specialization constants are replaced with default values.
  ImageImpl ImageWithReplacedSpecConsts = nullptr;
  // Original image where specialization constants are not replaced with
  // default values.
  ImageImpl OriginalImage = nullptr;
  // Used to track if any of the candidate images has specialization values
  // set.
  bool SpecConstsSet = false;
  for (auto &DeviceImage : MDeviceImages) {
    if (!DeviceImage.has_kernel(KernelID))
      continue;

    const auto DeviceImageImpl = detail::getSyclObjImpl(DeviceImage);
    SpecConstsSet |= DeviceImageImpl->is_any_specialization_constant_set();

    // Remember current image in corresponding variable depending on whether
    // specialization constants are replaced with default value or not.
    (DeviceImageImpl->specialization_constants_replaced_with_default()
         ? ImageWithReplacedSpecConsts
         : OriginalImage) = DeviceImageImpl;

    if (SpecConstsSet) {
      // If specialization constant is set in any of the candidate images
      // then we can't use ReplacedImage, so we select NativeImage if any or
      // we select OriginalImage and keep iterating in case there is an image
      // with native support.
      SelectedImage = OriginalImage;
      if (SelectedImage && SelectedImage->all_specialization_constant_native())
        break;
    } else {
      // For now select ReplacedImage but it may be reset if any of the
      // further device images has specialization constant value set. If after
      // all iterations specialization constant values are not set in any of
      // the candidate images then that will be the selected image.
      // Also we don't want to use ReplacedImage if device image has native
      // support.
      if (ImageWithReplacedSpecConsts &&
          !ImageWithReplacedSpecConsts->all_specialization_constant_native())
        SelectedImage = ImageWithReplacedSpecConsts;
      else
        // In case if we don't have or don't use ReplacedImage.
        SelectedImage = OriginalImage;
    }
  }
  return SelectedImage;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
