#include "vk/storage.hpp"

#include "vk/common.hpp"

namespace ve
{
Storage::Storage(const VulkanMainContext& vmc, VulkanCommandContext& vcc) : vmc(vmc), vcc(vcc)
{}

void Storage::destroy_buffer(uint32_t idx)
{
  if (buffers.at(idx).has_value())
  {
    buffers.at(idx).value().destruct();
    buffers.at(idx).reset();
  }
  else
  {
    std::cerr << "Trying to destroy already destroyed buffer!" << std::endl;
  }
}

void Storage::destroy_image(uint32_t idx)
{
  if (images.at(idx).has_value())
  {
    images.at(idx).value().destruct();
    images.at(idx).reset();
  }
  else
  {
    std::cerr << "Trying to destroy already destroyed image!" << std::endl;
  }
}

void Storage::destroy_buffer(const std::string& name)
{
  destroy_buffer(buffer_names.at(name));
}

void Storage::destroy_image(const std::string& name)
{
  destroy_image(image_names.at(name));
}

void Storage::clear()
{
  for (const std::pair<std::string, uint32_t>& buffer : buffer_names)
  {
    if (buffers[buffer.second].has_value())
    {
      std::cout << "WARNING: Buffer \"" << buffer.first << "\" not destroyed! Cleaning up..." << std::endl;
      buffers[buffer.second].value().destruct();
    }
  }
  buffers.clear();
  for (const std::pair<std::string, uint32_t>& image : image_names)
  {
    if (images[image.second].has_value())
    {
      std::cout << "WARNING: Image \"" << image.first << "\" not destroyed! Cleaning up..." << std::endl;
      images[image.second].value().destruct();
    }
  }
  images.clear();
  buffer_names.clear();
  image_names.clear();
}

Buffer& Storage::get_buffer(uint32_t idx)
{
  if (!buffers.at(idx).has_value()) VE_THROW("Trying to get already destroyed buffer!");
  return buffers.at(idx).value();
}

Image& Storage::get_image(uint32_t idx)
{
  if (!images.at(idx).has_value()) VE_THROW("Trying to get already destroyed image!");
  return images.at(idx).value();
}

Buffer& Storage::get_buffer_by_name(const std::string& name)
{
  if (!buffer_names.contains(name)) VE_THROW("Failed to find buffer with name: " + name);
  return get_buffer(buffer_names.at(name));
}

Image& Storage::get_image_by_name(const std::string& name)
{
  if (!image_names.contains(name)) VE_THROW("Failed to find image with name: " + name);
  return get_image(image_names.at(name));
}
} // namespace ve
