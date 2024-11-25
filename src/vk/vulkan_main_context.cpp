#include "vk/vulkan_main_context.hpp"

#include "vk/ve_log.hpp"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity, VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data)
{
  switch (message_severity)
  {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      std::cerr << "validation verbose" << std::endl;
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      std::cerr << "validation info" << std::endl;
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      std::cerr << "validation warning" << std::endl;
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      std::cerr << "validation error" << std::endl;
      break;
  }
  std::cout << VE_C_LBLUE << callback_data->pMessage << VE_C_WHITE << std::endl;
  return VK_FALSE;
}

namespace ve
{
void VulkanMainContext::construct(const uint32_t width, const uint32_t height)
{
  window = std::make_unique<Window>(width, height);
  instance.construct(window->get_required_extensions());
  surface = window->create_surface(instance.get());
  physical_device.construct(instance, surface);
  queue_family_indices = physical_device.get_queue_families(surface);
  logical_device.construct(physical_device, queue_family_indices, queues);
  create_vma_allocator();
  setup_debug_messenger();
}

void VulkanMainContext::destruct()
{
  vmaDestroyAllocator(va);
  instance.get().destroySurfaceKHR(surface);
  logical_device.destruct();
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) instance.get().getProcAddr("vkDestroyDebugUtilsMessengerEXT");
  func(instance.get(), debug_messenger, nullptr);
  instance.destruct();
  if (window) window->destruct();
}

std::vector<vk::SurfaceFormatKHR> VulkanMainContext::get_surface_formats() const
{
  return physical_device.get().getSurfaceFormatsKHR(surface);
}

std::vector<vk::PresentModeKHR> VulkanMainContext::get_surface_present_modes() const
{
  return physical_device.get().getSurfacePresentModesKHR(surface);
}

vk::SurfaceCapabilitiesKHR VulkanMainContext::get_surface_capabilities() const
{
  return physical_device.get().getSurfaceCapabilitiesKHR(surface);
}

const vk::Queue& VulkanMainContext::get_graphics_queue() const
{
  return queues.at(QueueIndex::Graphics);
}

const vk::Queue& VulkanMainContext::get_transfer_queue() const
{
  return queues.at(QueueIndex::Transfer);
}

const vk::Queue& VulkanMainContext::get_compute_queue() const
{
  return queues.at(QueueIndex::Compute);
}

const vk::Queue& VulkanMainContext::get_present_queue() const
{
  return queues.at(QueueIndex::Present);
}

void VulkanMainContext::create_vma_allocator()
{
  VmaAllocatorCreateInfo vaci{};
  vaci.instance = instance.get();
  vaci.physicalDevice = physical_device.get();
  vaci.device = logical_device.get();
  vaci.vulkanApiVersion = VK_API_VERSION_1_3;
  vaci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  vmaCreateAllocator(&vaci, &va);
}

void VulkanMainContext::setup_debug_messenger()
{
  vk::DebugUtilsMessengerCreateInfoEXT dumci;
  dumci.sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT;
  dumci.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
  dumci.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
  dumci.pfnUserCallback = debug_callback;
  debug_messenger = instance.get().createDebugUtilsMessengerEXT(dumci, nullptr, VULKAN_HPP_DEFAULT_DISPATCHER);
}
} // namespace ve
