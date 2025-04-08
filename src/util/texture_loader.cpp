#include "util/texture_loader.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/vulkan_command_context.hpp"
#include "stb/stb_image.h"

namespace ve {
TextureResourceImGui::TextureResourceImGui(const VulkanMainContext& vmc, Storage& storage) : vmc(vmc), storage(storage)
{}

void TextureResourceImGui::construct(const std::string& filePath) 
{
    int tex_width, tex_height, tex_channels;
    unsigned char* pixels = stbi_load(filePath.c_str(), &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
    if (!pixels) 
    {
        throw std::runtime_error("Failed to load texture image: " + filePath);
    }

    vk::ImageUsageFlags usage_flags = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc;
    
    std::vector<uint32_t> queue_family_indices = {vmc.queue_family_indices.transfer};
    
    // Create texture
    texture = storage.add_image("persistence diagram", pixels, static_cast<uint32_t>(tex_width), static_cast<uint32_t>(tex_height), false, 0, queue_family_indices, usage_flags);

    stbi_image_free(pixels);

    vk::DescriptorPoolSize pool_size;
    pool_size.descriptorCount = 1;

    vk::DescriptorPoolCreateInfo pool_info;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;

    VE_CHECK(vmc.logical_device.get().createDescriptorPool(&pool_info, nullptr, &descriptor_pool), "Failed to create descriptor pool");

    vk::DescriptorSetLayoutBinding layout_binding;
    layout_binding.binding = 0;
    layout_binding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    layout_binding.descriptorCount = 1;
    layout_binding.stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo layout_info;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &layout_binding;

    VE_CHECK(vmc.logical_device.get().createDescriptorSetLayout(&layout_info, nullptr, &descriptor_set_layout), "Failed to create descriptor set layout");

    vk::DescriptorSetAllocateInfo alloc_info;
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout;

    VE_CHECK(vmc.logical_device.get().allocateDescriptorSets(&alloc_info, &descriptor_set), "Failed to allocate descriptor set");

    vk::DescriptorImageInfo image_info;
    image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    image_info.imageView = storage.get_image(texture).get_view();
    image_info.sampler = storage.get_image(texture).get_sampler();

    vk::WriteDescriptorSet descriptor_write;
    descriptor_write.dstSet = descriptor_set;
    descriptor_write.dstBinding = 0;
    descriptor_write.dstArrayElement = 0;
    descriptor_write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    descriptor_write.descriptorCount = 1;
    descriptor_write.pImageInfo = &image_info;

    vmc.logical_device.get().updateDescriptorSets(1, &descriptor_write, 0, nullptr);
}

void TextureResourceImGui::destruct() 
{
    vmc.logical_device.get().destroyDescriptorSetLayout(descriptor_set_layout, nullptr);
    vmc.logical_device.get().destroyDescriptorPool(descriptor_pool, nullptr);
    storage.destroy_image(texture);
}

ImTextureID TextureResourceImGui::getImTextureID() const 
{
    return reinterpret_cast<ImTextureID>(VkDescriptorSet(descriptor_set));
}
}