#pragma once
#include <vulkan/vulkan.hpp>
#include <string>
#include <memory>
#include "vk/image.hpp"

namespace ve {

class VulkanMainContext;
class VulkanCommandContext;

struct TextureResource 
{
    std::unique_ptr<Image> texture;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
};

class TextureLoader 
{
public:
    static TextureResource loadTexture(const VulkanMainContext& vmc, VulkanCommandContext& vcc, const std::string& filePath);
    static void destroyTextureResource(const VulkanMainContext& vmc, TextureResource& resource);
};
}