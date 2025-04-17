// ui.cpp

#include "ui.hpp"
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_sdl3.h"

#include <iostream>
#include "threshold_cut.hpp"
#include "persistence.hpp"
#include "merge_tree.hpp"
#include "volume.hpp"
#include <cmath>
#include <limits>

namespace ve
{
UI::UI(const VulkanMainContext& vmc)
    : vmc(vmc), normalizationFactor(255.0f) // default; will be updated by the WorkContext
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

void UI::set_persistent_dots(const std::vector<ImVec2>& dots)
{
    persistentDots = dots;
}

void UI::set_normalization_factor(float nf)
{
    normalizationFactor = nf;
}

void UI::draw(vk::CommandBuffer& cb, AppState& app_state)
{
    // Start a new ImGui frame.
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

    // camera controls…
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Camera Settings");
        ImGui::DragFloat("Camera Speed", &app_state.move_speed, 10.0f, 0.0f, 100.0f);
    }
    ImGui::Separator();

    // persistent feature selection controls
    if (ImGui::CollapsingHeader("Persistent Feature Selection", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::SliderInt("Persistence Threshold", &app_state.persistence_threshold, 110000, 110300);
        if (ImGui::Button("Apply Persistence Threshold"))
        {
            app_state.apply_persistence_threshold = true;
        }
        ImGui::SameLine();
        ImGui::Text("Current Threshold: %d", app_state.persistence_threshold);

        ImGui::SliderInt("Target Level", &app_state.target_level, 0, 300);
        if (ImGui::Button("Apply Target Level"))
        {
            app_state.apply_target_level = true;
            if (merge_tree)
                merge_tree->set_target_level(app_state.target_level);
        }
        ImGui::SameLine();
        ImGui::Text("Current Level: %d", app_state.target_level);
    }
    ImGui::Separator();

    // filtration mode controls
    if (ImGui::CollapsingHeader("Filtration Mode", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int currentMode = 0; // 0 for LowerStar, 1 for UpperStar
        const char* modeOptions[] = { "Lower Star", "Upper Star" };
        if (ImGui::Combo("Mode", &currentMode, modeOptions, IM_ARRAYSIZE(modeOptions)))
        {
            app_state.filtration_mode = (currentMode == 0) ? FiltrationMode::LowerStar : FiltrationMode::UpperStar;
        }
        if (ImGui::Button("Apply Filtration Mode"))
        {
            app_state.apply_filtration_mode = true;
        }
        ImGui::SameLine();
        ImGui::Text("Current mode: %s", (currentMode == 0) ? "Lower Star" : "Upper Star");
    }
    ImGui::PushItemWidth(80.0f);
    ImGui::Separator();
    ImGui::Text((std::to_string(app_state.time_diff * 1000) + " ms; FPS: " + std::to_string(1.0 / app_state.time_diff)).c_str());
    ImGui::Text("'G': Show/Hide UI");
    ImGui::End();

   ImGui::Begin("Persistence Diagram");

    if (persistence_pairs) {
        std::cout << "DEBUG: persistence_pairs size = " << persistence_pairs->size() << "\n";
        ImGui::Text("Persistence pairs count: %d", int(persistence_pairs->size()));
    } else {
        std::cout << "DEBUG: persistence_pairs pointer is NULL\n";
        ImGui::Text("Persistence pairs pointer is NULL");
    }

    const float base_dot_radius = 7.0f;

    // calculate effective dot radius (scale with zoom)
    ImGuiIO& io = ImGui::GetIO();
    diagramZoom += io.MouseWheel * 0.1f;
    if (diagramZoom < 0.1f) 
        diagramZoom = 0.1f;
    ImVec2 base_size(500, 500);
    ImVec2 diagram_size(base_size.x * diagramZoom, base_size.y * diagramZoom);
    float effective_dot_radius = base_dot_radius * diagramZoom;

    // draw the persistence diagram using an ImageButton for an exact clickable area
    if (ImGui::ImageButton("Diagram", persistence_texture_ID, diagram_size, ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0)))
    {
        ImVec2 image_pos = ImGui::GetItemRectMin();
        ImVec2 image_max = ImGui::GetItemRectMax();
        ImVec2 mouse_pos = ImGui::GetMousePos();

        // Check bounds (should be automatically true with ImageButton).
        if (mouse_pos.x < image_pos.x || mouse_pos.y < image_pos.y ||
            mouse_pos.x > image_max.x || mouse_pos.y > image_max.y)
        {
            std::cout << "DEBUG: Click is outside the diagram bounds. Ignoring click.\n";
        }
        else
        {
            std::cout << "DEBUG: Mouse pos: (" << mouse_pos.x << ", " << mouse_pos.y << ")\n";
            std::cout << "DEBUG: Diagram bounds: (" << image_pos.x << ", " << image_pos.y 
                      << ") -> (" << image_max.x << ", " << image_max.y << ")\n";
            std::cout << "DEBUG: Effective dot radius = " << effective_dot_radius << " px\n";

            bool hitFound = false;
            PersistencePair selectedPair(0, 0);

            // iterate over all persistence pairs
            if (persistence_pairs)
            {
                int debugCount = 0;
                for (const PersistencePair& pair : *persistence_pairs)
                {
                    float pair_x = image_pos.x + (static_cast<float>(pair.birth) / 255.0f)*diagram_size.x;
                    float pair_y = image_pos.y + ((255.0f - static_cast<float>(pair.death)) / 255.0f)*diagram_size.y;
                    
                    if (debugCount < 3)
                    {
                        std::cout << "DEBUG: Candidate " << debugCount 
                                  << " (birth, death): (" << pair.birth << ", " << pair.death 
                                  << ") -> Screen Pos (" << pair_x << ", " << pair_y << ")\n";
                        debugCount++;
                    }
                    float dx = mouse_pos.x - pair_x;
                    float dy = mouse_pos.y - pair_y;
                    float dist = std::sqrt(dx * dx + dy * dy);
                    if (dist <= effective_dot_radius)
                    {
                        selectedPair = pair;
                        hitFound = true;
                        break;
                    }
                }
            }
            else
            {
                std::cout << "DEBUG: persistence_pairs pointer is NULL!\n";
            }

            if (hitFound)
            {
                std::cout << "DEBUG: Selected persistence pair: (birth=" << selectedPair.birth
                          << ", death=" << selectedPair.death << ") within effective dot radius.\n";
                if (on_pair_selected)
                    on_pair_selected(selectedPair);
            }
            else
            {
                std::cout << "DEBUG: No persistence pair hit (click not within effective dot radius " 
                          << effective_dot_radius << " px).\n";
            }
        }
    }

    ImGui::End();
    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
} 
} // namespace ve
