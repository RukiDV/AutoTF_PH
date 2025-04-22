#include "ui.hpp"
#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_sdl3.h"
#include <cmath>
#include <limits>
#include <implot.h> 

#include <iostream>
#include "threshold_cut.hpp"
#include "persistence.hpp"
#include "merge_tree.hpp"
#include "volume.hpp"

namespace ve {

UI::UI(const VulkanMainContext& vmc) : vmc(vmc), normalization_factor(255.0f)
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
    ImPlot::CreateContext();
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
    ImPlot::DestroyContext();
    vmc.logical_device.get().destroyDescriptorPool(imgui_pool);
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
    cache_dirty = true; 
}

void UI::set_persistence_texture(ImTextureID tex)
{
    persistence_texture_ID = tex;
}

void UI::set_on_pair_selected(const std::function<void(const PersistencePair&)>& callback)
{
    on_pair_selected = callback;
}

void UI::set_on_range_applied(std::function<void(const std::vector<PersistencePair>&)> cb)
{ 
    on_range_applied = std::move(cb);
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

    // camera controls
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
    ImGui::Text("VOLUME: %.4f ms", app_state.device_timings[DeviceTimer::VOLUME]);
    ImGui::Text("UI: %.4f ms", app_state.device_timings[DeviceTimer::UI]);
    ImGui::End();

    // persistence diagram
    ImGui::Begin("Persistence Diagram", nullptr,ImGuiWindowFlags_HorizontalScrollbar);
    {
        if (!persistence_pairs || persistence_pairs->empty())
        {
            ImGui::Text("No persistence pairs to display");
        }
        else
        {
            int N = int(persistence_pairs->size());
            // reset button
            ImGui::Text("Total pairs: %d", N);

            // initialize slider to max on first show
            static bool first_time = true;
            if (first_time)
            {
                max_points_to_show = N;
                first_time = false;
            }
            max_points_to_show = std::min(max_points_to_show, N);
            if (ImGui::Button("Reset Controls"))
            {
                show_dots = true;
                max_points_to_show = N;
                birth_range[0] = 0.0f;
                birth_range[1] = 255.0f;
                death_range[0] = 0.0f;
                death_range[1] = 255.0f;
                persistence_range[0] = 0.0f;
                persistence_range[1] = 255.0f;
                diagram_zoom = 1.0f;
                marker_size = 5.0f;
                cache_dirty = true;
                range_active = false;
            }
            ImGui::SameLine();
            // controls
            ImGui::Checkbox("Show Dots", &show_dots);
            ImGui::SliderInt("Max Points", &max_points_to_show, 1, N);
            ImGui::SliderFloat2("Birth Range", birth_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat2("Death Range", death_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat2( "Persistence Range", persistence_range, 0.0f, 255.0f, "%.0f" );
            ImGui::SliderFloat("Zoom", &diagram_zoom, 0.1f, 3.0f, "%.2f");
            ImGui::SliderFloat("Marker Size",&marker_size, 1.0f, 20.0f, "%.1f");

            ImGuiIO& io = ImGui::GetIO();
            blink_timer += io.DeltaTime; // accumulate seconds
            bool blink_on = fmodf(blink_timer, 0.5f) < 0.25f;

            // begin ImPlot
            if (ImPlot::BeginPlot("##PD", ImVec2(500 * diagram_zoom, 500 * diagram_zoom)))
            {
                ImPlot::SetupAxes("Birth","Death");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, 255, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 255, ImPlotCond_Always);

                // adjust zoom via mouse wheel
                ImGuiIO& io = ImGui::GetIO();
                if (ImPlot::IsPlotHovered() && io.MouseWheel != 0.0f)
                {
                    diagram_zoom = std::clamp(diagram_zoom + io.MouseWheel * 0.2f, 0.1f, 10.0f);
                }

                // prepare lists of indices that pass the range and persistence filter
                std::vector<int> idxs;
                idxs.reserve(N);
                for (int i = 0; i < N; ++i)
                {
                    const auto &p = (*persistence_pairs)[i];
                    float birth = float(p.birth);
                    float death = float(p.death);
                    float pers  = death - birth;
                    if( birth  >= birth_range[0]    && birth  <= birth_range[1] &&
                        death  >= death_range[0]    && death  <= death_range[1] &&
                        pers   >= persistence_range[0] && pers   <= persistence_range[1] )
                    {
                        idxs.push_back(i);
                        if( idxs.size() >= max_points_to_show ) break;
                    }
                }
                
                if (ImGui::Button("Apply Range Filter"))
                {
                    range_active = true;
                    std::vector<PersistencePair> filtered;
                    filtered.reserve(idxs.size());
                    for (int i : idxs)
                        filtered.push_back((*persistence_pairs)[i]);
                    if (on_range_applied)
                        on_range_applied(filtered);
                }
                ImGui::SameLine();
                
                // compute some constant pixel padding inside the plot
                const float pad_px = 10.0f * diagram_zoom;  // 10px on each side
                ImDrawList* dl = ImPlot::GetPlotDrawList();
                ImVec2 plot_pos = ImPlot::GetPlotPos();
                ImVec2 plot_size = ImPlot::GetPlotSize();
                const float inner_w = plot_size.x - 2 * pad_px;
                const float inner_h = plot_size.y - 2 * pad_px;

                ImPlot::PushPlotClipRect();

                if (show_dots)
                {
                    // rebuild cache if needed
                    if (cache_dirty)
                    {
                        xs.resize(N);
                        ys.resize(N);
                        pers.resize(N);
                        float maxP = 0.0f;
                        for (int i = 0; i < N; ++i)
                        {
                            auto &p = (*persistence_pairs)[i];
                            xs[i]   = double(p.birth);
                            ys[i]   = double(p.death);
                            float d = float(p.death - p.birth);
                            pers[i] = d;
                            if (d > maxP) maxP = d;
                        }
                        for (int i = 0; i < N; ++i)
                            pers[i] = pers[i] / (maxP > 0 ? maxP : 1.0f);
                        cache_dirty = false;
                    }

                    int count = std::min(max_points_to_show, N);
                    dot_pos.clear();
                    dot_pos.reserve(count);
                    for (int k = 0; k < (int)idxs.size(); ++k)
                    {
                        int i = idxs[k];
                        auto &p = (*persistence_pairs)[i];
                        float fx = float(p.birth) / 255.0f;
                        float fy = float(p.death) / 255.0f;
                        ImVec2 pos = {plot_pos.x + pad_px + fx * inner_w, plot_pos.y + pad_px + (1.0f - fy) * inner_h};
                        dot_pos.push_back(pos);

                        // color by persistence (hue mapping)
                        float hue = (1.0f - pers[i]) * 0.66f;
                        float r,g,b;
                        ImGui::ColorConvertHSVtoRGB(hue,1,1,r,g,b);
                        ImU32 col = IM_COL32(int(r*255), int(g*255), int(b*255), 255);

                        dl->AddCircleFilled(pos, marker_size, col);
                        
                        // if this was clicked, draw an outline ring
                        if (k == selected_idx && blink_on)
                            dl->AddCircle(pos, marker_size + 2.0f, selected_color, 16, 2.0f);
                    }

                    // click to select
                    if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(0) && !range_active)
                    {
                        ImVec2 m = ImGui::GetIO().MousePos;
                        float best_r2 = marker_size * marker_size;
                        int   best_i  = -1;
                        for (int i = 0; i < (int)dot_pos.size(); ++i)
                        {
                            float dx = m.x - dot_pos[i].x;
                            float dy = m.y - dot_pos[i].y;
                            float d2 = dx * dx + dy * dy;
                            if (d2 < best_r2)
                            {
                                best_r2 = d2;
                                best_i = i;
                            }
                        }
                        if (best_i >= 0)
                        {
                            selected_idx = best_i;
                            if (on_pair_selected)
                            on_pair_selected((*persistence_pairs)[best_i]);
                        }
                    }
                }
                ImPlot::PopPlotClipRect();
                ImPlot::EndPlot();

                if (selected_idx >= 0)
                {
                    auto &p = (*persistence_pairs)[selected_idx];
                    ImGui::Text("Selected Pair: (%u , %u)", p.birth, p.death);
                }
            }
        }
        ImGui::End();
    }

    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
}
} // namespace ve
