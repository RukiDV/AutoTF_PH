#include "util/texture_loader.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/vulkan_command_context.hpp"
#include "stb/stb_image.h"

namespace ve {

TextureResource TextureLoader::loadTexture(const VulkanMainContext& vmc, VulkanCommandContext& vcc, const std::string& filePath) 
{
    TextureResource resource{};
    
    // load image
    int tex_width, tex_height, tex_channels;
    unsigned char* pixels = stbi_load(filePath.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
    if (!pixels) 
    {
        throw std::runtime_error("Failed to load texture image: " + filePath);
    }

    vk::ImageUsageFlags usage_flags = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc;
    
    std::vector<uint32_t> queue_family_indices = {vmc.queue_family_indices.transfer};
    
    // Create texture
    resource.texture = std::make_unique<Image>(vmc, vcc, pixels, static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height), true, 0, queue_family_indices, usage_flags);
    
    stbi_image_free(pixels);

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;

    if (vkCreateDescriptorPool(vmc.logical_device.get(), &pool_info, nullptr, &resource.descriptor_pool) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor pool");
    }

    VkDescriptorSetLayoutBinding layout_binding{};
    layout_binding.binding = 0;
    layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &layout_binding;

    if (vkCreateDescriptorSetLayout(vmc.logical_device.get(), &layout_info, nullptr, &resource.descriptor_set_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create descriptor set layout");
    }

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = resource.descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &resource.descriptor_set_layout;

    if (vkAllocateDescriptorSets(vmc.logical_device.get(), &alloc_info, &resource.descriptor_set) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to allocate descriptor set");
    }

    VkDescriptorImageInfo image_info{};
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView = resource.texture->get_view();
    image_info.sampler = resource.texture->get_sampler();

    VkWriteDescriptorSet descriptor_write{};
    descriptor_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_write.dstSet = resource.descriptor_set;
    descriptor_write.dstBinding = 0;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptor_write.descriptorCount = 1;
    descriptor_write.pImageInfo = &image_info;

    vkUpdateDescriptorSets(vmc.logical_device.get(), 1, &descriptor_write, 0, nullptr);

    return resource;
}

void TextureLoader::destroyTextureResource(const VulkanMainContext& vmc, TextureResource& resource) 
{
    if (resource.descriptor_set_layout)
    {
        vkDestroyDescriptorSetLayout(vmc.logical_device.get(), resource.descriptor_set_layout, nullptr);
    }
    if (resource.descriptor_pool)
    {
        vkDestroyDescriptorPool(vmc.logical_device.get(), resource.descriptor_pool, nullptr);
    }
    if (resource.texture)
    {
        resource.texture->destruct();
    }
}
}