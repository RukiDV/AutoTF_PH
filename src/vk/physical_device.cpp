#include "vk/physical_device.hpp"

#include <unordered_set>

#include "vk/ve_log.hpp"

namespace ve
{
void PhysicalDevice::construct(const Instance& instance, const std::optional<vk::SurfaceKHR>& surface)
{
  const std::vector<const char*> required_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  const std::vector<const char*> optional_extensions{};
  extensions_handler.add_extensions(required_extensions, true);
  extensions_handler.add_extensions(optional_extensions, false);

  std::vector<vk::PhysicalDevice> physical_devices = instance.get_physical_devices();
  std::unordered_set<uint32_t> suitable_p_devices;
  for (uint32_t i = 0; i < 1; ++i)
  {
    if (is_device_suitable(i, physical_devices[i], surface))
    {
      suitable_p_devices.insert(i);
    }
  }
  if (suitable_p_devices.size() > 1)
  {
    uint32_t pd_idx = 0;
    do
    {
      std::cout << "Select one of the suitable GPUs by typing the number" << std::endl;
      std::cin >> pd_idx;
    } while (!suitable_p_devices.contains(pd_idx));
    physical_device = physical_devices[pd_idx];
  }
  else if (suitable_p_devices.size() == 1)
  {
    std::cerr << "Only one suitable GPU. Using this one." << std::endl;
    physical_device = physical_devices[*(suitable_p_devices.begin())];
  }
  else
  {
    VE_THROW("No suitable GPUs found!");
  }
  vk::PhysicalDeviceProperties pdp = physical_device.getProperties();
  std::cerr << "GPU: " << std::string(pdp.deviceName.data()) << std::endl;
  extensions_handler.remove_missing_extensions();
}

vk::PhysicalDevice PhysicalDevice::get() const
{
  return physical_device;
}

const std::vector<const char*>& PhysicalDevice::get_extensions() const
{
  return extensions_handler.get_extensions();
}

const std::vector<const char*>& PhysicalDevice::get_missing_extensions()
{
  return extensions_handler.get_missing_extensions();
}

bool PhysicalDevice::is_device_suitable(uint32_t idx, const vk::PhysicalDevice p_device, const std::optional<vk::SurfaceKHR>& surface)
{
  vk::PhysicalDeviceProperties pdp = p_device.getProperties();
  vk::PhysicalDeviceFeatures p_device_features = p_device.getFeatures();
  std::vector<vk::ExtensionProperties> available_extensions = p_device.enumerateDeviceExtensionProperties();
  std::vector<const char*> avail_ext_names;
  for (const auto& ext : available_extensions) avail_ext_names.push_back(ext.extensionName);
  std::cout << "    " << idx << " " << pdp.deviceName << " ";
  int32_t missing_extensions = extensions_handler.check_extension_availability(avail_ext_names);
  if (missing_extensions == -1 || !p_device_features.samplerAnisotropy || (surface.has_value() && extensions_handler.find_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME) && !is_swapchain_supported(p_device, surface.value())))
  {
    std::cout << "(not suitable)\n";
    return false;
  }
  std::cout << "(suitable, " << missing_extensions << " missing optional extensions)\n";
  return true;
}

bool PhysicalDevice::is_swapchain_supported(const vk::PhysicalDevice p_device, const vk::SurfaceKHR& surface) const
{
  std::vector<vk::SurfaceFormatKHR> f = p_device.getSurfaceFormatsKHR(surface);
  std::vector<vk::PresentModeKHR> pm = p_device.getSurfacePresentModesKHR(surface);
  return !f.empty() && !pm.empty();
}

} // namespace ve
