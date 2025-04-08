#pragma once
#include "vk/common.hpp"
#include <string>
#include "vk/storage.hpp"
#include "imgui/imgui.h"

namespace ve {

class VulkanMainContext;
class VulkanCommandContext;

class TextureResourceImGui
{
public:
    TextureResourceImGui(const VulkanMainContext& vmc, Storage& storage);
    void construct(const std::string& filePath);
    void destruct();
    ImTextureID getImTextureID() const;

private:
    const VulkanMainContext& vmc;
    Storage& storage;
    uint32_t texture;
    vk::DescriptorPool descriptor_pool;
    vk::DescriptorSetLayout descriptor_set_layout;
    vk::DescriptorSet descriptor_set;
};
}