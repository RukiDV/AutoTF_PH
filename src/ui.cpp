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

void UI::set_on_multi_selected(const std::function<void(const std::vector<PersistencePair>&)>& cb)
{
    on_multi_selected = cb;
}

void UI::set_on_brush_selected(const std::function<void(const std::vector<PersistencePair>&)>& cb)
{
    on_brush_selected = cb;
}

void UI::set_gradient_persistence_pairs(const std::vector<PersistencePair>* gp)
{
    gradient_pairs = gp;
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
    ImGui::Begin("Persistence Diagram", nullptr, ImGuiWindowFlags_HorizontalScrollbar);
    {
        // pick overall visualization
        static int viewType = 0;
        const char* viewNames[] = { "Persistence", "Barcode", "Merge Tree" };
        ImGui::Combo("Visualization", &viewType, viewNames, IM_ARRAYSIZE(viewNames));
        ImGui::Separator();

        // pick scalar vs gradient persistence
        static int pd_mode = 0;
        ImGui::Text("Persistence Pairs Mode:"); ImGui::SameLine();
        ImGui::RadioButton("Scalar persistence",  &pd_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Gradient persistence", &pd_mode, 1);
        ImGui::Separator();

        // choose which set to draw
        const auto* draw_pairs = (pd_mode == 1 && gradient_pairs) ? gradient_pairs : persistence_pairs;

        if (!draw_pairs || draw_pairs->empty())
        {
            ImGui::Text("No persistence pairs to display");
            ImGui::End();
            return;
        }

        int N = int(draw_pairs->size());
        ImGui::Text("Total pairs: %d", N);
        ImGui::Separator();

        // automatic initial highlight of most persistent feature
        static bool initial_feature_highlighted = false;
        if (!initial_feature_highlighted && draw_pairs && !draw_pairs->empty())
        {
            // find the pair with max persistence (death - birth)
            auto it = std::max_element(draw_pairs->begin(), draw_pairs->end(), [](auto &a, auto &b)
            {
                return (a.death - a.birth) < (b.death - b.birth);
            });
            PersistencePair most = *it;

            if (viewType == 0)
            {
                if (on_pair_selected) on_pair_selected(most);
                selected_idx = int(std::distance(draw_pairs->begin(), it));
            }
            else if (viewType == 1)
            {
                if (on_range_applied) on_range_applied({ most });
            }

            initial_feature_highlighted = true;
        }

        // persistence diagram view
        if (viewType == 0)
        {
            // display mode: iso‐surface vs volume‐highlight
            static int displayMode = 1; 
            ImGui::Text("Display Mode:"); ImGui::SameLine();
            ImGui::RadioButton("Iso-surface", &displayMode, 0);ImGui::SameLine();
            ImGui::RadioButton("Volume-highlight", &displayMode, 1);
            ImGui::Separator();
            app_state.display_mode = displayMode;

            // reset when you switch iso vs highlight
            static int lastMode = 1;
            if (displayMode != lastMode)
            {
                range_active = false;
                selected_idx = -1;
                multi_selected_idxs.clear();
                multi_selected_cols.clear();
                lastMode = displayMode;
            }

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
                birth_range[0] = death_range[0] = persistence_range[0] = 0.0f;
                birth_range[1] = death_range[1] = persistence_range[1] = 255.0f;
                diagram_zoom = 1.0f;
                marker_size = 5.0f;
                cache_dirty = true;
                range_active = false;
                multi_selected_idxs.clear();
                multi_selected_cols.clear();
            }
            ImGui::SameLine();
            ImGui::Checkbox("Show Dots", &show_dots);
            ImGui::SliderInt("Max Points", &max_points_to_show, 1, N);
            ImGui::SliderFloat2("Birth Range", birth_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat2("Death Range", death_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat2("Persistence Range", persistence_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat("Zoom", &diagram_zoom, 0.1f, 3.0f, "%.2f");
            ImGui::SliderFloat("Marker Size", &marker_size, 1.0f, 20.0f, "%.1f");

            ImGuiIO& io = ImGui::GetIO();
            blink_timer += io.DeltaTime;
            bool blink_on = fmodf(blink_timer, 0.5f) < 0.25f;

            if (ImPlot::BeginPlot("##PD", ImVec2(500 * diagram_zoom, 500 * diagram_zoom)))
            {
                ImPlot::SetupAxes("Birth","Death");
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, 255, ImPlotCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 255, ImPlotCond_Always);

                // zoom with mouse wheel
                if (ImPlot::IsPlotHovered() && io.MouseWheel != 0.0f)
                    diagram_zoom = std::clamp(diagram_zoom + io.MouseWheel * 0.2f, 0.1f, 10.0f);

                // filter index list
                std::vector<int> idxs;
                idxs.reserve(N);
                for (int i = 0; i < N; ++i)
                {
                    const auto &p = (*draw_pairs)[i];
                    float birth = float(p.birth);
                    float death = float(p.death);
                    float pers = death - birth;
                    if (birth >= birth_range[0] && birth <= birth_range[1] && death >= death_range[0] && death <= death_range[1] && pers >= persistence_range[0] && pers <= persistence_range[1])
                    {
                        idxs.push_back(i);
                        if ((int)idxs.size() >= max_points_to_show) break;
                    }
                }

                if (ImGui::Button("Apply Range Filter"))
                {
                    range_active = true;
                    std::vector<PersistencePair> filtered;
                    filtered.reserve(idxs.size());
                    for (int i : idxs)
                        filtered.push_back((*draw_pairs)[i]);
                    if (on_range_applied)
                        on_range_applied(filtered);
                }
                ImGui::SameLine();

                // draw dots
                const float pad = 10.0f * diagram_zoom;
                ImDrawList* dl = ImPlot::GetPlotDrawList();
                ImVec2 origin = ImPlot::GetPlotPos();
                ImVec2 plot_size = ImPlot::GetPlotSize();
                float inner_w = plot_size.x - 2 * pad;
                float inner_h = plot_size.y - 2 * pad;
                ImPlot::PushPlotClipRect();

                if (show_dots)
                {
                    // rebuild cache once
                    if (cache_dirty)
                    {
                        xs.resize(N); ys.resize(N); pers.resize(N);
                        float maxP = 0.0f;
                        for (int i = 0; i < N; ++i)
                        {
                            auto &p = (*draw_pairs)[i];
                            xs[i] = double(p.birth);
                            ys[i] = double(p.death);
                            pers[i] = float(p.death - p.birth);
                            maxP = std::max(maxP, pers[i]);
                        }
                        for (int i = 0; i < N; ++i)
                            pers[i] /= (maxP > 0.0f ? maxP : 1.0f);
                        cache_dirty = false;
                    }

                    // plot each dot
                    dot_pos.clear();
                    dot_pos.reserve(idxs.size());
                    for (int k = 0; k < (int)idxs.size(); ++k)
                    {
                        int i = idxs[k];
                        auto &p = (*draw_pairs)[i];
                        float fx = p.birth / 255.0f;
                        float fy = p.death / 255.0f;
                        ImVec2 pos {origin.x + pad + fx * inner_w, origin.y + pad + (1.0f - fy) * inner_h};
                        dot_pos.push_back(pos);

                        float hue = (1.0f - pers[i]) * 0.66f;
                        float r,g,b;
                        ImGui::ColorConvertHSVtoRGB(hue, 1, 1, r, g, b);
                        dl->AddCircleFilled(pos, marker_size, IM_COL32(int(r*255),int(g*255),int(b*255),255));

                        if (blink_on && k == selected_idx)
                            dl->AddCircle(pos, marker_size + 2.0f, selected_color, 16, 2.0f);
                    }

                    // brush on drag
                    if (ImPlot::IsPlotHovered() && ImGui::IsMouseDragging(0))
                    {
                        if (!brush_active)
                        {
                            brush_active = true;
                            brush_start = io.MouseClickedPos[0];
                        }
                        brush_end = io.MousePos;
                    }

                    // finish brush
                    if (brush_active && ImGui::IsMouseReleased(0))
                    {
                        brush_active = false;
                        float dx = brush_end.x - brush_start.x;
                        float dy = brush_end.y - brush_start.y;
                        float r2 = dx*dx + dy*dy;
                        std::vector<PersistencePair> brush_sel;
                        for (size_t i = 0; i < dot_pos.size(); ++i)
                        {
                            float ddx = dot_pos[i].x - brush_start.x;
                            float ddy = dot_pos[i].y - brush_start.y;
                            if (ddx * ddx + ddy * ddy <= r2)
                                brush_sel.push_back((*draw_pairs)[idxs[i]]);
                        }
                        if (!brush_sel.empty() && on_brush_selected)
                            on_brush_selected(brush_sel);
                    }

                    // click select
                    if (!brush_active && ImPlot::IsPlotHovered() && ImGui::IsMouseReleased(0))
                    {
                        ImVec2 m = io.MousePos;
                        float best_r2 = marker_size * marker_size;
                        int best_i = -1;
                        for (int i = 0; i < (int)dot_pos.size(); ++i)
                        {
                            float dx = m.x - dot_pos[i].x;
                            float dy = m.y - dot_pos[i].y;
                            if (dx * dx + dy * dy < best_r2)
                            {
                                best_r2 = dx * dx + dy * dy;
                                best_i = i;
                            }
                        }
                        if (best_i >= 0)
                        {
                            // ctrl+click toggles multi-select
                            if (io.KeyCtrl)
                            {
                                auto it = std::find(multi_selected_idxs.begin(), multi_selected_idxs.end(), best_i);
                                if (it == multi_selected_idxs.end())
                                {
                                    multi_selected_idxs.push_back(best_i);
                                    float hue = float(multi_selected_idxs.size()-1) / 6.0f;
                                    float r,g,b;ImGui::ColorConvertHSVtoRGB(hue, 1, 1, r, g, b);
                                    multi_selected_cols.push_back(IM_COL32(int(r * 255), int(g * 255), int(b * 255), 255));
                                } else
                                {
                                    size_t idx = std::distance(multi_selected_idxs.begin(), it);
                                    multi_selected_idxs.erase(it);
                                    multi_selected_cols.erase(multi_selected_cols.begin() + idx);
                                }
                                if (on_multi_selected)
                                {
                                    std::vector<PersistencePair> sel;
                                    for (int k : multi_selected_idxs)
                                        sel.push_back((*draw_pairs)[idxs[k]]);
                                    on_multi_selected(sel);
                                }
                            } else
                            {
                                multi_selected_idxs.clear();
                                multi_selected_cols.clear();
                                selected_idx = best_i;
                                PersistencePair p = (*draw_pairs)[idxs[best_i]];
                                if (displayMode == 0)
                                {
                                    if (on_pair_selected) on_pair_selected(p);
                                } else
                                {
                                    if (on_range_applied) on_range_applied({ p });
                                }
                            }
                        }
                    }

                    // draw brush overlay
                    if (brush_active)
                    {
                        float rad = std::sqrt((brush_end.x - brush_start.x)*(brush_end.x - brush_start.x) + (brush_end.y - brush_start.y)*(brush_end.y - brush_start.y)
                        );
                        dl->AddCircle(brush_start, rad, IM_COL32(255,255,0,150), 64, 2.0f);
                    }

                    // draw multi-select overlays
                    for (size_t m = 0; m < multi_selected_idxs.size(); ++m)
                    {
                        int k = multi_selected_idxs[m];
                        ImVec2 pos = dot_pos[k];
                        dl->AddCircleFilled(pos, marker_size + 1.5f, multi_selected_cols[m]);
                        dl->AddCircle(pos, marker_size + 3.0f, IM_COL32(255,255,255,200), 16, 2.0f);
                    }
                }

                ImPlot::PopPlotClipRect();
                ImPlot::EndPlot();

                if (selected_idx >= 0)
                {
                    auto &p = (*draw_pairs)[selected_idx];
                    ImGui::Text("Selected Pair: (%u , %u)", p.birth, p.death);
                }
            }
        }
        // barcode view
        else if (viewType == 1)
        {
            static bool show_barcodes = true;
            ImGui::Checkbox("Show Bars", &show_barcodes);
            ImGui::Separator();

            static float min_persistence = 0.0f;
            static int top_k = 10;
            static float barcode_zoom = 1.0f;
            static int selected_bar_rank = -1;
            static ImVec2 click_start;
            static bool rect_select_active = false;
            static ImVec2 rect_start, rect_end;
            static std::vector<int> multi_ranks;

            int maxBars = draw_pairs ? int(draw_pairs->size()) : 1;

            if (ImGui::Button("Reset Controls"))
            {
                min_persistence = 0.0f;
                top_k = 1;
                barcode_zoom = 1.0f;
                selected_bar_rank = -1;
                multi_ranks.clear();
                rect_select_active = false;
            }
            ImGui::Separator();
            ImGui::SliderFloat("Min Persistence", &min_persistence, 0.0f, 255.0f, "%.0f");
            ImGui::Text("Top K Bars:"); ImGui::SameLine();
            ImGui::SliderInt("##top_k_slider", &top_k, 1, maxBars);
            ImGui::SameLine(); ImGui::PushItemWidth(100);
            ImGui::InputInt("##top_k_input", &top_k);
            ImGui::PopItemWidth();
            top_k = std::clamp(top_k, 1, maxBars);
            ImGui::Separator();
            ImGui::SliderFloat("Barcode Zoom", &barcode_zoom, 0.1f, 5.0f, "%.2f");
            ImGui::Text("Scroll to zoom; Ctrl-click or drag to multi-select");

            if (!draw_pairs || draw_pairs->empty())
            {
                ImGui::Text("No persistence pairs to display");
            }
            else
            {
                // collect & sort eligible bars and find max persistence
                std::vector<std::pair<float,int>> lengths;
                float maxP = 0.0f;
                for (int i = 0; i < int(draw_pairs->size()); ++i)
                {
                    float L = float((*draw_pairs)[i].death - (*draw_pairs)[i].birth);
                    if (L >= min_persistence)
                    {
                        lengths.emplace_back(L, i);
                        maxP = std::max(maxP, L);
                    }
                }
                std::sort(lengths.begin(), lengths.end(), [](auto &a, auto &b){ return a.first > b.first; });
                int display_count = std::min(top_k, int(lengths.size()));
                ImVec2 plot_size(-1, 300 * barcode_zoom);

                if (ImPlot::BeginPlot("##Barcode", plot_size))
                {
                    ImPlot::SetupAxes("Value","Bar Rank");
                    ImPlot::SetupAxisLimits(ImAxis_X1, 0, 255, ImPlotCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 0, float(display_count+1), ImPlotCond_Always);

                    ImGuiIO& io = ImGui::GetIO();
                    if (ImPlot::IsPlotHovered() && io.MouseWheel != 0.0f)
                        barcode_zoom = std::clamp(barcode_zoom + io.MouseWheel * 0.25f, 0.1f, 10.0f);

                        // record click start
                        if (ImPlot::IsPlotHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
                            click_start = io.MouseClickedPos[0];

                        // start marquee
                        if (ImPlot::IsPlotHovered() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
                        {
                            rect_select_active = true;
                            rect_start = click_start;
                            rect_end   = io.MousePos;
                        }

                        // on release: marquee or click
                        if (ImPlot::IsPlotHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
                        {
                            // marquee 
                            if (rect_select_active)
                            {
                                rect_select_active = false;
                                multi_ranks.clear();
                                ImVec2 rmin{std::min(rect_start.x, rect_end.x), std::min(rect_start.y, rect_end.y)};
                                ImVec2 rmax{std::max(rect_start.x, rect_end.x), std::max(rect_start.y, rect_end.y)};
                                ImVec2 origin = ImPlot::GetPlotPos();
                                ImVec2 size = ImPlot::GetPlotSize();

                                for (int rank = 0; rank < display_count; ++rank)
                                {
                                    int idx = lengths[rank].second;
                                    auto &p = (*draw_pairs)[idx];
                                    // screen endpoints
                                    ImVec2 s0{origin.x + (p.birth  / 255.0f) * size.x, origin.y + (1.0f - float(display_count-rank)/(display_count+1)) * size.y
                                    };
                                    ImVec2 s1{origin.x + (p.death  / 255.0f) * size.x, s0.y
                                    };
                                    if ((s0.x>=rmin.x && s0.x<=rmax.x && s0.y>=rmin.y && s0.y<=rmax.y) || (s1.x>=rmin.x && s1.x<=rmax.x && s1.y>=rmin.y && s1.y<=rmax.y))
                                    {
                                        multi_ranks.push_back(rank);
                                    }
                                }
                                if (!multi_ranks.empty() && on_multi_selected)
                                {
                                    std::vector<PersistencePair> sel;
                                    for (int rank : multi_ranks)
                                        sel.push_back((*draw_pairs)[ lengths[rank].second ]);
                                    on_multi_selected(sel);
                                }
                            }
                            // click or ctrl+click
                            else
                            {
                                ImPlotPoint mp = ImPlot::GetPlotMousePos();
                                int cr = display_count - int(std::round(mp.y));
                                cr = std::clamp(cr, 0, display_count-1);

                                if (io.KeyCtrl)
                                {
                                    auto it = std::find(multi_ranks.begin(), multi_ranks.end(), cr);
                                    if (it == multi_ranks.end())
                                        multi_ranks.push_back(cr);
                                    else
                                        multi_ranks.erase(it);

                                    if (!multi_ranks.empty() && on_multi_selected)
                                    {
                                        std::vector<PersistencePair> sel;
                                        for (int rank : multi_ranks)
                                            sel.push_back((*draw_pairs)[ lengths[rank].second ]);
                                        on_multi_selected(sel);
                                    }
                                }
                                else
                                {
                                    multi_ranks.clear();
                                    selected_bar_rank = cr;
                                    int idx = lengths[cr].second;
                                    PersistencePair clicked{ (*draw_pairs)[idx].birth, (*draw_pairs)[idx].death };
                                    if (on_range_applied)
                                        on_range_applied({ clicked });
                                }
                            }
                        }

                        // draw marquee rectangle overlay
                        if (rect_select_active)
                        {
                            ImDrawList* dl = ImPlot::GetPlotDrawList();
                            dl->AddRectFilled(rect_start, rect_end, IM_COL32(255,255,0,80));
                            dl->AddRect(rect_start, rect_end, IM_COL32(255,255,0,200), 0.0f, 0, 2.0f);
                        }

                        // draw bars: pink if selected/multi, else persistence hue
                        if (show_barcodes)
                        {
                            for (int rank = 0; rank < display_count; ++rank)
                            {
                                int idx = lengths[rank].second;
                                auto &p = (*draw_pairs)[idx];
                                double xs[2] = {double(p.birth), double(p.death)};
                                double ys[2] = {double(display_count - rank), double(display_count - rank)};
                                char buf[32];
                                std::snprintf(buf, sizeof(buf), "##bar%02d", idx);

                                bool is_sel = (rank == selected_bar_rank) || (std::find(multi_ranks.begin(), multi_ranks.end(), rank) != multi_ranks.end());

                                ImVec4 col;
                                if (is_sel)
                                    col = ImVec4(1.0f, 0.4f, 0.7f, 1.0f); // pink
                                else {
                                    float hue = (1.0f - (lengths[rank].first / (maxP>0?maxP:1.0f))) * 0.66f;
                                    float r,g,b; ImGui::ColorConvertHSVtoRGB(hue,1,1,r,g,b);
                                    col = ImVec4(r,g,b,1.0f);
                                }

                                float weight = is_sel ? 3.0f * barcode_zoom : 1.5f * barcode_zoom;
                                ImPlot::SetNextLineStyle(col, weight);
                                ImPlot::PlotLine(buf, xs, ys, 2);
                            }
                        }
                    ImPlot::EndPlot();
                }
            }
        }

        // merge tree view
        else
        {
            ImGui::Text("Merge Tree view todo…");
        }
    }
    ImGui::End();

    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
}
} // namespace ve
