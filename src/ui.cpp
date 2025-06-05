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

UI::UI(const VulkanMainContext& vmc) : vmc(vmc), normalization_factor(255.0f), current_pd_mode(0)
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

void UI::set_merge_tree(MergeTree* mt)
{
    merge_tree = mt;
}

void UI::set_on_merge_mode_changed(const std::function<void(int)>& cb)
{
    on_merge_mode_changed = cb;
}

void UI::mark_merge_tree_dirty() { mt_dirty = true; }

// change to accept ramp arg
void UI::set_on_brush_selected_gradient(const std::function<void(const std::vector<std::pair<PersistencePair, float>>&, int)>& cb)
{
    on_brush_selected_gradient = cb;
}

void UI::set_on_highlight_selected(const std::function<void(const std::vector<std::pair<PersistencePair,float>>&,int)>& cb)
{
    on_highlight_selected = cb;
}

void UI::set_on_diff_selected(const std::function<void(const PersistencePair&, const PersistencePair&)>& cb)
{ 
    on_diff_selected = cb;
}
void UI::set_on_intersect_selected(const std::function<void(const PersistencePair&, const PersistencePair&)>& cb)
{
    on_intersect_selected = cb;
}

void UI::set_on_union_selected(const std::function<void(const PersistencePair&, const PersistencePair&)>& cb)
{
    on_union_selected = cb;
}


void UI::clear_selection()
{
    selected_idx = -1;
    range_active = false;
    last_highlight_hits.clear();
    multi_selected_idxs.clear();
    multi_selected_cols.clear();
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
        ImGui::Text("Persistence Pairs Mode:"); ImGui::SameLine();
        bool pd_changed = false;
        static int pd_mode = 0;
        if (ImGui::RadioButton("Scalar persistence", &pd_mode, 0)) pd_changed = true;
        ImGui::SameLine();
        if (ImGui::RadioButton("Gradient persistence", &pd_mode, 1)) pd_changed = true;

        if (pd_changed)
        {
            if (on_merge_mode_changed)
                on_merge_mode_changed(pd_mode);

            current_pd_mode = pd_mode;
            selected_idx = -1;
            last_highlight_hits.clear();
            multi_selected_idxs.clear();
            multi_selected_cols.clear();
            range_active = false;
        }

        ImGui::Separator();
 
        ImGui::Text("Highlight Appearance");
        ImGui::SliderFloat("Highlight Opacity", &highlight_opacity, 0.0f, 1.0f, "%.2f");
        
        static float prev_opacity = highlight_opacity;
        if (highlight_opacity != prev_opacity)
        {
            prev_opacity = highlight_opacity;

            for (auto &h : last_highlight_hits)
            {
                h.second = highlight_opacity;
            }

            if (!last_highlight_hits.empty() && on_highlight_selected)
                on_highlight_selected(last_highlight_hits, selected_ramp);
        } 

        const char* ramp_names[] = { "HSV (Blue->Red)", "Viridis", "Plasma", "Magma", "Inferno", "Custom" };
        ImGui::Combo("Color Ramp", &selected_ramp, ramp_names, IM_ARRAYSIZE(ramp_names));

        static int prev_ramp = selected_ramp;
        if (selected_ramp != prev_ramp)
        {
            prev_ramp = selected_ramp;
            if (!last_highlight_hits.empty() && on_highlight_selected)
                on_highlight_selected(last_highlight_hits, selected_ramp);
        }

        if (selected_ramp == RAMP_CUSTOM)
        {
            ImGui::ColorEdit4("Start Color", &custom_start_color.x);
            ImGui::ColorEdit4("End Color",   &custom_end_color.x);
            ImGui::SliderFloat("Opacity Falloff", &custom_opacity_falloff, 0.0f, 1.0f, "%.2f");

            static ImVec4 prev_c0 = custom_start_color;
            static ImVec4 prev_c1 = custom_end_color;
            static float  prev_f  = custom_opacity_falloff;
            if (custom_start_color.x != prev_c0.x || custom_start_color.y != prev_c0.y || custom_start_color.z != prev_c0.z || custom_start_color.w != prev_c0.w || custom_end_color.x   != prev_c1.x   || custom_end_color.y   != prev_c1.y   || custom_end_color.z != prev_c1.z   || custom_end_color.w   != prev_c1.w || custom_opacity_falloff != prev_f)
            {
                prev_c0 = custom_start_color;
                prev_c1 = custom_end_color;
                prev_f  = custom_opacity_falloff;
                if (!last_highlight_hits.empty() && on_highlight_selected)
                    on_highlight_selected(last_highlight_hits, selected_ramp);
            }
        }

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
                brush_outer_mult = 1.0f;
                brush_inner_ratio = 0.7f;  
            }
            ImGui::SameLine();
            ImGui::Checkbox("Show Dots", &show_dots);
            ImGui::SliderInt("Max Points", &max_points_to_show, 1, N);
            ImGui::SliderFloat2("Birth Range", birth_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat2("Death Range", death_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat2("Persistence Range", persistence_range, 0.0f, 255.0f, "%.0f");
            ImGui::SliderFloat("Zoom", &diagram_zoom, 0.1f, 3.0f, "%.2f");
            ImGui::SliderFloat("Marker Size", &marker_size, 1.0f, 20.0f, "%.1f");
            ImGui::SliderFloat("Brush Size Multiplier", &brush_outer_mult, 0.1f, 2.0f, "%.2f");
            ImGui::SliderFloat("Inner Radius Ratio",   &brush_inner_ratio,   0.0f, 1.0f, "%.2f");
            ImGui::Separator();

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

                if (selected_idx >= (int)idxs.size())
                    selected_idx = -1;

                // remove any ctrl+click selections that are now invalid
                for (auto it = multi_selected_idxs.begin(); it != multi_selected_idxs.end();)
                {
                    if (*it >= (int)idxs.size())
                    {
                        size_t color_idx = std::distance(multi_selected_idxs.begin(), it);
                        it = multi_selected_idxs.erase(it);
                        multi_selected_cols.erase(multi_selected_cols.begin() + color_idx);
                    }
                    else
                    {
                        ++it;
                    }
                }

                if (ImGui::Button("Apply Range Filter"))
                {
                    range_active = true;
                    std::vector<PersistencePair> filtered;
                    filtered.reserve(idxs.size());
                    for (int i : idxs)
                        filtered.push_back((*draw_pairs)[i]);
                    
                    std::vector<std::pair<PersistencePair,float>> hits;
                    hits.reserve(filtered.size());
                    for (auto &p : filtered)
                        hits.emplace_back(p, 1.0f);

                    last_highlight_hits = hits;
                    if (!hits.empty() && on_highlight_selected)
                        on_highlight_selected(hits, selected_ramp); 
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
                        ImVec2 pos = ImVec2(origin.x + pad + fx * inner_w, origin.y + pad + (1.0f - fy) * inner_h);
                        dot_pos.push_back(pos);

                        float tval = pers[i];

                        // ramp‐based color lookup
                        float cr=0.0f, cg=0.0f, cb=0.0f;
                        switch (selected_ramp)
                        {
                            case RAMP_HSV:
                            {
                                float hue = (1.0f - tval) * 0.66f;
                                ImGui::ColorConvertHSVtoRGB(hue, 1.0f, 1.0f, cr, cg, cb);
                                break;
                            }
                            case RAMP_VIRIDIS:
                            {
                                glm::vec3 col = viridis(tval);
                                cr = col.x; cg = col.y; cb = col.z;
                                break;
                            }
                            case RAMP_PLASMA:
                            {
                                glm::vec3 col = plasma(tval);
                                cr = col.x; cg = col.y; cb = col.z;
                                break;
                            }
                            case RAMP_MAGMA:
                            {
                                glm::vec3 col = magma(tval);
                                cr = col.x; cg = col.y; cb = col.z;
                                break;
                            }
                            case RAMP_INFERNO:
                            {
                                glm::vec3 col = inferno(tval);
                                cr = col.x; cg = col.y; cb = col.z;
                                break;
                            }
                            case RAMP_CUSTOM:
                            {
                                ImVec4 a = custom_start_color;
                                ImVec4 bcol = custom_end_color;
                                cr = a.x + tval * (bcol.x - a.x);
                                cg = a.y + tval * (bcol.y - a.y);
                                cb = a.z + tval * (bcol.z - a.z);
                                break;
                            }
                            default:
                            {
                                ImGui::ColorConvertHSVtoRGB(tval, 1.0f, 1.0f, cr, cg, cb);
                                break;
                            }
                        }
                        cr = glm::clamp(cr, 0.0f, 1.0f);
                        cg = glm::clamp(cg, 0.0f, 1.0f);
                        cb = glm::clamp(cb, 0.0f, 1.0f);

                        // draw the dot
                        dl->AddCircleFilled(pos, marker_size, IM_COL32(int(cr*255), int(cg*255), int(cb*255), 255));

                        // if it is a very dark color draw a faint white outline
                        float lum = 0.2126f*cr + 0.7152f*cg + 0.0722f*cb;
                        if (lum < 0.05f)
                        {
                            dl->AddCircle(pos, marker_size + 0.2f, IM_COL32(255,255,255,100), 12, 1.0f);
                        }

                        if (blink_on && k == selected_idx)
                        {
                            dl->AddCircle(pos, marker_size + 2.0f, selected_color, 16, 2.0f);
                        }
                    }

                    // feathered brush
                    if (ImPlot::IsPlotHovered() && ImGui::IsMouseDragging(0))
                    {
                        if (!brush_active)
                        {
                            brush_active = true;
                            brush_start = io.MouseClickedPos[0];
                        }
                        brush_end = io.MousePos;
                    }

                    if (brush_active && ImGui::IsMouseReleased(0))
                    {
                        brush_active = false;
                        float dx = brush_end.x - brush_start.x;
                        float dy = brush_end.y - brush_start.y;
                        // apply user multiplier
                        float raw_r2 = dx*dx + dy*dy;
                        float max_r2 = raw_r2 * brush_outer_mult * brush_outer_mult;
                        float inner_r2 = max_r2 * brush_inner_ratio * brush_inner_ratio;

                        std::vector<std::pair<PersistencePair, float>> brush_sel;
                        brush_sel.reserve(dot_pos.size());
                        for (size_t i = 0; i < dot_pos.size(); ++i)
                        {
                            float ddx = dot_pos[i].x - brush_start.x;
                            float ddy = dot_pos[i].y - brush_start.y;
                            float dist2 = ddx*ddx + ddy*ddy;
                            if (dist2 <= max_r2)
                            {
                                float opacity = (dist2 <= inner_r2) ? 1.0f : 1.0f - (std::sqrt(dist2) - std::sqrt(inner_r2)) / (std::sqrt(max_r2) - std::sqrt(inner_r2));
                                float final_op = opacity * highlight_opacity;
                                brush_sel.emplace_back((*draw_pairs)[idxs[i]], final_op); 
                            }
                        }
                        last_highlight_hits = brush_sel;
                        if (!brush_sel.empty() && on_highlight_selected)
                            on_highlight_selected(brush_sel, selected_ramp);
                    }

                    if (brush_active)
                    {
                        float raw_r = std::sqrt((brush_end.x - brush_start.x)*(brush_end.x - brush_start.x) + (brush_end.y - brush_start.y)*(brush_end.y - brush_start.y));
                        float r   = raw_r * brush_outer_mult;
                        float ri  = r * brush_inner_ratio;

                        dl->AddCircle(brush_start, r,  IM_COL32(255,255,0,150), 64, 2.0f);
                        dl->AddCircle(brush_start, ri, IM_COL32(255,255,0,255), 64, 2.0f);
                    }

                    // click select & multi-select
                    if (!brush_active && ImPlot::IsPlotHovered() && ImGui::IsMouseReleased(0))
                    {
                        ImVec2 m = io.MousePos;
                        float best_r2 = marker_size * marker_size;
                        int best_i = -1;
                        for (int i = 0; i < (int)dot_pos.size(); ++i)
                        {
                            float dx = m.x - dot_pos[i].x, dy = m.y - dot_pos[i].y;
                            if (dx*dx + dy*dy < best_r2)
                            {
                                best_r2 = dx*dx + dy*dy;
                                best_i = i;
                            }
                        }
                        if (best_i >= 0)
                        {
                            // determine the base opacity from the slider
                            float base_opacity = highlight_opacity;

                            std::vector<std::pair<PersistencePair,float>> hits;

                            if (io.KeyCtrl)
                            {
                                // toggle and collect multi‐select indices
                                auto it = std::find(multi_selected_idxs.begin(), multi_selected_idxs.end(), best_i);
                                if (it == multi_selected_idxs.end())
                                {
                                    multi_selected_idxs.push_back(best_i);
                                    float hue = float(multi_selected_idxs.size()-1) / 6.0f;
                                    float r,g,b;
                                    ImGui::ColorConvertHSVtoRGB(hue, 1, 1, r, g, b);
                                    multi_selected_cols.push_back(IM_COL32(int(r*255),int(g*255),int(b*255),255));
                                }
                                else
                                {
                                    size_t idx = std::distance(multi_selected_idxs.begin(), it);
                                    multi_selected_idxs.erase(it);
                                    multi_selected_cols.erase(multi_selected_cols.begin() + idx);
                                }
                                // build hits with chosen base opacity
                                for (int k : multi_selected_idxs)
                                    hits.emplace_back((*draw_pairs)[idxs[k]], base_opacity);
                            }
                            else
                            {
                                multi_selected_idxs.clear();
                                multi_selected_cols.clear();
                                hits.emplace_back((*draw_pairs)[idxs[best_i]], base_opacity);
                                selected_idx = best_i;
                            }
                            last_highlight_hits = hits;
                            if (!hits.empty() && on_highlight_selected)
                                on_highlight_selected(hits, selected_ramp);
                        }
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
                ImGui::NewLine(); ImGui::Spacing();

                // if exactly two points have been ctrl-multi-clicked, show “Apply Diff”
                if (multi_selected_idxs.size() == 2)
                {
                    const PersistencePair &p1 = (*draw_pairs)[idxs[multi_selected_idxs[0]]];
                    const PersistencePair &p2 = (*draw_pairs)[idxs[multi_selected_idxs[1]]];
                    bool need_update = false;

                    // dropdown for set operations
                    static int selected_set_op = 0;
                    const char* set_op_names[] = { "Difference", "Intersection", "Union" };
                    if (ImGui::Combo("Set Operation", &selected_set_op, set_op_names, IM_ARRAYSIZE(set_op_names)))
                    {
                        need_update = true;
                    }
                    ImGui::Separator();

                    // color editor and checkboxes for each operation
                    if (selected_set_op == 0) // difference
                    {
                        // A \ B
                        if (ImGui::Checkbox("Show A \\ B", &diff_enabled))
                            need_update = true;

                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays only the voxels that are in A but not in B.");

                        if (diff_enabled)
                        {
                            ImGui::SameLine();
                            ImGui::Text("Color:");
                            ImGui::SameLine();
                            if (ImGui::ColorEdit4("##diff_color", &diff_color.x, ImGuiColorEditFlags_AlphaBar))
                                need_update = true;
                        }
                        ImGui::Separator();
                    }
                    else if (selected_set_op == 1) // intersection
                    {
                        // A ∩ B
                        if (ImGui::Checkbox("Show A and B", &intersect_enabled_common))
                            need_update = true;

                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays the voxels that are both in A and B.");

                        ImGui::SameLine();
                        ImGui::Text("Color:");
                        ImGui::SameLine();
                        if (ImGui::ColorEdit4("##intersect_color_common", &intersect_color_common.x, ImGuiColorEditFlags_AlphaBar))
                            need_update = true;
                        ImGui::Separator();

                        // A \ B
                        if (ImGui::Checkbox("Show A \\ B", &intersect_enabled_Aonly))
                            need_update = true;
                        
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays only the voxels that are only in A.");

                        ImGui::SameLine();
                        ImGui::Text("Color:");
                        ImGui::SameLine();
                        if (ImGui::ColorEdit4("##intersect_color_Aonly", &intersect_color_Aonly.x, ImGuiColorEditFlags_AlphaBar))
                            need_update = true;
                        ImGui::Separator();

                        // B \ A
                        if (ImGui::Checkbox("Show B \\ A", &intersect_enabled_Bonly))
                            need_update = true;

                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays only the voxels that are only in B.");

                        ImGui::SameLine();
                        ImGui::Text("Color:");
                        ImGui::SameLine();
                        if (ImGui::ColorEdit4("##intersect_color_Bonly", &intersect_color_Bonly.x, ImGuiColorEditFlags_AlphaBar))
                            need_update = true;

                    }
                    else if (selected_set_op == 2) // union
                    {
                        // A \ B
                        if (ImGui::Checkbox("Show A \\ B", &union_enabled_Aonly))
                            need_update = true;
                        
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays all voxels that are only in A.");

                        ImGui::SameLine();
                        ImGui::Text("Color:");
                        ImGui::SameLine();
                        if (ImGui::ColorEdit4("##union_color_Aonly", &union_color_Aonly.x, ImGuiColorEditFlags_AlphaBar))
                            need_update = true;
                        ImGui::Separator();

                        // B \ A
                        if (ImGui::Checkbox("Show B \\ A", &union_enabled_Bonly))
                            need_update = true;

                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays all voxels that are only in B.");

                        ImGui::SameLine();
                        ImGui::Text("Color:");
                        
                        ImGui::SameLine();
                        if (ImGui::ColorEdit4("##union_color_Bonly", &union_color_Bonly.x, ImGuiColorEditFlags_AlphaBar))
                            need_update = true;
                        ImGui::Separator();

                        // A ∩ B
                        if (ImGui::Checkbox("Show A or B", &union_enabled_common))
                            need_update = true;

                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Displays all voxels that are in A or B.");

                        ImGui::SameLine();
                        ImGui::Text("Color:");
                        ImGui::SameLine();
                        if (ImGui::ColorEdit4("##union_color_common", &union_color_common.x, ImGuiColorEditFlags_AlphaBar))
                            need_update = true;
                    }

                    ImGui::Separator();

                    if (need_update)
                    {
                        switch (selected_set_op)
                        {
                            case 0: // difference
                                if (on_diff_selected) on_diff_selected(p1, p2);
                                break;
                            case 1: // intersection
                                if (on_intersect_selected) on_intersect_selected(p1, p2);
                                break;
                            case 2: // union
                                if (on_union_selected) on_union_selected(p1, p2);
                                break;
                        }
                    }
                }

                ImGui::NewLine(); ImGui::Spacing();

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
                                else
                                {
                                    float hue = (1.0f - (lengths[rank].first / (maxP>0?maxP:1.0f))) * 0.66f;
                                    float r,g,b;
                                    ImGui::ColorConvertHSVtoRGB(hue,1,1,r,g,b);
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
        else if (viewType == 2)
        {
            if (!merge_tree)
            {
                ImGui::Text("No merge tree loaded");
                return;
            }

            // scalar vs gradient toggle
            static int last_mt_mode = -1;
            static int mt_mode = 0; // 0=scalar, 1=gradient
            ImGui::Text("Merge Tree Source:");
            ImGui::SameLine();
            ImGui::RadioButton("Scalar",   &mt_mode, 0);
            ImGui::SameLine();
            ImGui::RadioButton("Gradient", &mt_mode, 1);
            ImGui::Separator();
            
            if (mt_mode != last_mt_mode)
            {
                last_mt_mode = mt_mode;
                mt_dirty = true;
                if (on_merge_mode_changed)
                    on_merge_mode_changed(mt_mode);
            }

            // depth and persistence controls
            ImGui::TextColored(ImVec4(0.8f,0.8f,0.2f,1.0f), "Merge Tree Controls");
            static int depth_level = app_state.target_level;
            ImGui::SliderInt("Target Depth", &depth_level, 0, 10);
            if (ImGui::Button("Apply Depth"))
            {
                app_state.target_level = depth_level;
                merge_tree->set_target_level(depth_level);
                app_state.apply_target_level = true;
                mt_dirty = true;
            }
            ImGui::SameLine(); ImGui::Text("Current: %d", app_state.target_level);

            static int persist_thr = app_state.persistence_threshold;
            ImGui::SliderInt("Persistence Thr", &persist_thr, 0, 255);
            if (ImGui::Button("Apply Thr"))
            {
                app_state.persistence_threshold = persist_thr;
                merge_tree->set_persistence_threshold(persist_thr);
                app_state.apply_persistence_threshold = true;
                mt_dirty = true;

                std::vector<PersistencePair> survivors;
                for (auto &p : *persistence_pairs)
                {
                    if ((int)p.death - (int)p.birth >= persist_thr)
                        survivors.push_back(p);
                }

                if (on_range_applied)
                    on_range_applied(survivors);
            }
            ImGui::SameLine(); ImGui::Text("Current: %d", app_state.persistence_threshold);

            ImGui::Separator();

            // zoom & pan
            static float zoom = 1.0f;
            static ImVec2 pan{0,0};
            ImGui::SliderFloat("Tree Zoom", &zoom, 0.5f, 3.0f, "%.2f");
            ImGui::Text("Drag on canvas to pan");

            // child canvas (ImGui clears background)
            ImGui::BeginChild("##MergeTreeCanvas", ImVec2(300,0), false, ImGuiWindowFlags_NoScrollWithMouse);
            ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
            ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
            ImDrawList* dl = ImGui::GetWindowDrawList();

            // pan handling
            ImGui::InvisibleButton("pan_canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft);
            if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                pan.x += ImGui::GetIO().MouseDelta.x;
                pan.y += ImGui::GetIO().MouseDelta.y;
            }

            // rebuild caches only when dirty
            if (mt_dirty)
            {
                mt_dirty = false;
                mt_edges.clear();
                mt_nodes.clear();

                // DFS + prune
                struct Frame { MergeTreeNode* n; int d; };
                std::vector<Frame> stack;
                std::vector<MergeTreeNode*> nodes;
                stack.reserve(512);
                nodes.reserve(512);

                for (auto &kv : merge_tree->get_all_nodes())
                    if (kv.second->parent == nullptr)
                        stack.push_back({kv.second,0});

                while (!stack.empty())
                {
                    auto f = stack.back(); stack.pop_back();
                    auto *n = f.n; int d = f.d;
                    if (d > app_state.target_level) continue;
                    if ((int)n->death - (int)n->birth < app_state.persistence_threshold
                        && n->children.empty()) continue;
                    n->depth = d;
                    nodes.push_back(n);
                    for (auto *c : n->children)
                        stack.push_back({c, d+1});
                }

                // compute max birth
                uint32_t maxB = 1;
                for (auto *n : nodes) maxB = std::max(maxB, n->birth);

                // layout positions and fill caches
                std::unordered_map<MergeTreeNode*,ImVec2> pos;
                pos.reserve(nodes.size());
                for (auto *n : nodes)
                {
                    float fx = float(n->birth)/float(maxB);
                    ImVec2 P = ImVec2(canvas_p0.x + pan.x + fx * canvas_sz.x * zoom, canvas_p0.y + pan.y + n->depth * 50.0f * zoom);
                    pos[n] = P;
                    mt_nodes.emplace_back(P, n->id);
                }
                for (auto *n : nodes)
                    for (auto *c : n->children)
                        mt_edges.emplace_back(pos[n], pos[c]);
            }

            // draw from cache
            const float edgeTh = 3.0f * zoom;
            for (auto &e : mt_edges)
                dl->AddLine(e.first, e.second, IM_COL32(200,200,120,255), edgeTh);

            for (auto &nd : mt_nodes)
            {
                dl->AddCircleFilled(nd.first, 5.0f * zoom, IM_COL32(100,200,100,255));
                char buf[16];
                std::snprintf(buf,sizeof(buf), "%u", nd.second);
                dl->AddText(
                ImGui::GetFont(),
                ImGui::GetFontSize(),
                ImVec2(nd.first.x + 7*zoom, nd.first.y - 7 * zoom),
                IM_COL32(240,240,240,255),
                buf
                );
            }
            ImGui::EndChild();
        }
    }
    ImGui::End();

    ImGui::EndFrame();
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
}
} // namespace ve