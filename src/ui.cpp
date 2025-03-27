#include "ui.hpp"
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_sdl3.h"
#include <iostream>

namespace ve
{
UI::UI(const VulkanMainContext& vmc) : vmc(vmc)
{}

void UI::construct(VulkanCommandContext& vcc, const RenderPass& render_pass, uint32_t frames)
{
  std::vector<vk::DescriptorPoolSize> pool_sizes =
  {
    { vk::DescriptorType::eSampler, 1000 },
    { vk::DescriptorType::eCombinedImageSampler, 1000 },
    { vk::DescriptorType::eSampledImage, 1000 },
    { vk::DescriptorType::eStorageImage, 1000 },
    { vk::DescriptorType::eUniformTexelBuffer, 1000 },
    { vk::DescriptorType::eStorageTexelBuffer, 1000 },
    { vk::DescriptorType::eUniformBuffer, 1000 },
    { vk::DescriptorType::eStorageBuffer, 1000 },
    { vk::DescriptorType::eUniformBufferDynamic, 1000 },
    { vk::DescriptorType::eStorageBufferDynamic, 1000 },
    { vk::DescriptorType::eInputAttachment, 1000 }
  };

  vk::DescriptorPoolCreateInfo dpci{};
  dpci.sType = vk::StructureType::eDescriptorPoolCreateInfo;
  dpci.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
  dpci.maxSets = 1000;
  dpci.poolSizeCount = pool_sizes.size();
  dpci.pPoolSizes = pool_sizes.data();

  imgui_pool = vmc.logical_device.get().createDescriptorPool(dpci);

  ImGui::CreateContext();
  ImGui_ImplSDL3_InitForVulkan(vmc.window->get());
  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.Instance = vmc.instance.get();
  init_info.PhysicalDevice = vmc.physical_device.get();
  init_info.Device = vmc.logical_device.get();
  init_info.Queue = vmc.get_graphics_queue();
  init_info.DescriptorPool = imgui_pool;
  init_info.RenderPass = render_pass.get();
  init_info.Subpass = 0;
  init_info.MinImageCount = frames;
  init_info.ImageCount = frames;
  init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

  ImGui_ImplVulkan_Init(&init_info);
  ImGui::StyleColorsDark();
}

void UI::destruct()
{
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();
  vmc.logical_device.get().destroyDescriptorPool(imgui_pool);
}

void UI::set_merge_tree(MergeTree* merge_tree)
{
  this->merge_tree = merge_tree;
}

void UI::set_transfer_function(TransferFunction* transfer_function)
{
  this->transfer_function = transfer_function;
}

void UI::set_volume(const Volume* volume)
{
  this->volume = volume;
}

void UI::set_persistence_pairs(const std::vector<PersistencePair>* pairs)
{
  this->persistence_pairs = pairs;
}

void UI::draw_merge_tree_node(const MergeTreeNode* node)
{
    if (node == nullptr) return;

    ImGui::PushID(node->id);
    if (ImGui::TreeNode((void*)(intptr_t)node->id, "Node %d (Birth: %d, Death: %d)", node->id, node->birth, node->death))
    {
        for (const auto& child : node->children)
        {
            draw_merge_tree_node(child);
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void UI::draw(vk::CommandBuffer& cb, AppState& app_state)
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplSDL3_NewFrame();
  ImGui::NewFrame();
  ImGui::Begin("AutoTF_PH");
  if (ImGui::CollapsingHeader("Navigation"))
  {
      ImGui::Text("'W'A'S'D'Q'E': movement");
      ImGui::Text("Mouse_L || Arrow-Keys: panning");
      ImGui::Text("'+'-': change movement speed");
      ImGui::Text("'G': Show/Hide UI");
      ImGui::Text("'F1': Screenshot");
  }
  ImGui::Separator();

  // camera
  if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
  {
      ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Camera Settings");
      ImGui::DragFloat("Camera Speed", &app_state.move_speed, 10.0f, 0.0f, 100.0f);
  }
  ImGui::Separator();

  // target level settings
  if (ImGui::CollapsingHeader("Target Level", ImGuiTreeNodeFlags_DefaultOpen))
  {
      int max_level = 100;
      ImGui::SliderInt("Set Level", &app_state.target_level, 0, max_level);
      if (ImGui::Button("Apply Target Level"))
      {
          app_state.apply_target_level = true;
          if (merge_tree)
              merge_tree->set_target_level(app_state.target_level);
      }
      ImGui::SameLine();
      ImGui::Text("Current: %d", app_state.target_level);
  }
  ImGui::Separator();

  // transfer function settings
  if (ImGui::CollapsingHeader("Transfer Function", ImGuiTreeNodeFlags_DefaultOpen))
  {
      if (ImGui::Button("Apply Histogram TF"))
      {
          app_state.apply_histogram_tf = true;
      }
      if (ImGui::Button("Apply Hybrid TF"))
      {
          app_state.apply_histogram_ph_tf = true;
      }
      ImGui::SliderInt("PH Threshold", &app_state.ph_threshold, 0, 100);
  }
  ImGui::Separator();

  // persistence threshold settings
  if (ImGui::CollapsingHeader("Persistence", ImGuiTreeNodeFlags_DefaultOpen))
  {
      ImGui::SliderInt("Persistence Threshold", &app_state.persistence_threshold, 0, 1000);
      if (ImGui::Button("Apply Persistence Threshold"))
      {
          std::cout << "Threshold button pressed!" << std::endl;
          if (merge_tree)
          {
              merge_tree->set_persistence_threshold(app_state.persistence_threshold);
          }
      }
      ImGui::SameLine();
      ImGui::Text("Current: %d", app_state.persistence_threshold);
  }
  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.0, 1.0, 0.0, 1.0), "Camera");
  ImGui::DragFloat("Camera speed", &app_state.move_speed, 10.0f, 0.0f, 100.0f);
  ImGui::PushItemWidth(80.0f);
  ImGui::Separator();
  ImGui::Text((std::to_string(app_state.time_diff * 1000) + " ms; FPS: " + std::to_string(1.0 / app_state.time_diff)).c_str());
  ImGui::Text("'G': Show/Hide UI");
  ImGui::End();
  ImGui::EndFrame();

  ImGui::Render();
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
}
} // namespace ve
