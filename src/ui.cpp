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

namespace ve {

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

    // show count
    int N = persistence_pairs ? int(persistence_pairs->size()) : 0;
    ImGui::Text("Persistence pairs count: %d", N);

    // controls for zoom & dot size
    static float diagramZoom = 1.0f;
    static float dotRadius   = 5.0f;
    ImGui::SliderFloat("Zoom",      &diagramZoom, 0.1f, 3.0f, "%.2f");
    ImGui::SliderFloat("Dot Radius",&dotRadius,   1.0f, 20.0f, "%.1f");

    // reserve a square canvas
    ImVec2 canvasSize = ImVec2(500.0f * diagramZoom, 500.0f * diagramZoom);
    ImGui::InvisibleButton("canvas", canvasSize);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 origin = ImGui::GetItemRectMin();       // top‑left of our canvas

    // draw diagonal
    ImU32 diagColor = IM_COL32(200,50,50,200);
    ImVec2 p_bot_left = ImVec2(origin.x,            origin.y + canvasSize.y);
    ImVec2 p_top_right = ImVec2(origin.x + canvasSize.x, origin.y);
    dl->AddLine(p_bot_left, p_top_right, diagColor, 1.0f);

    // draw all dots & record their screen positions, colouring by persistence
    static std::vector<ImVec2> dotPos;
    dotPos.clear();

    if (persistence_pairs && !persistence_pairs->empty())
    {
        // compute maximum persistence once
        float maxPers = 0.0f;
        for (auto &p : *persistence_pairs)
            maxPers = std::max(maxPers, float(p.death - p.birth));
        if (maxPers < 1e-6f) maxPers = 1.0f;

        for (size_t i = 0; i < persistence_pairs->size(); ++i)
        {
            const auto &p = (*persistence_pairs)[i];
            float bx = float(p.birth) / 255.0f;
            float dy = float(p.death) / 255.0f;

            ImVec2 pos {
            origin.x + bx     * canvasSize.x,
            origin.y + (1-dy) * canvasSize.y
            };
            dotPos.push_back(pos);

            // map persistence to hue [0=red … 0.66=blue]
            float persNorm = (p.death - p.birth) / maxPers;    // 0…1
            float hue      = (1.0f - persNorm) * 0.66f;        // invert so high persistence = red
            float r,g,b;
            ImGui::ColorConvertHSVtoRGB(hue, 1.0f, 1.0f, r, g, b);

            ImU32 col = IM_COL32(
                int(r * 255.0f),
                int(g * 255.0f),
                int(b * 255.0f),
                255
            );

            dl->AddCircleFilled(pos, dotRadius, col);
        }

    }

    // hit‑test on click
    if (ImGui::IsItemClicked())
    {
      ImVec2 m = ImGui::GetIO().MousePos;
      for (int i = 0; i < (int)dotPos.size(); ++i)
      {
        float dx = m.x - dotPos[i].x;
        float dy = m.y - dotPos[i].y;
        if (dx*dx + dy*dy <= dotRadius*dotRadius)
        {
          // Got it!
          PersistencePair hit = (*persistence_pairs)[i];
          std::cout << "DEBUG: Hit persistence pair " << i 
                    << " = (" << hit.birth << "," << hit.death << ")\n";
          if (on_pair_selected) 
            on_pair_selected(hit);
          break;
        }
      }
    }

    ImGui::End();
    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
} 
} // namespace ve
