#pragma once

#include "app_state.hpp"
#include "vk/render_pass.hpp"
#include "vk/vulkan_main_context.hpp"
#include "vk/vulkan_command_context.hpp"
#include "merge_tree.hpp"
#include "transfer_function.hpp"
#include <functional>
#include "imgui.h"
#include <vector>
#include "colormaps.hpp"

namespace ve
{
class UI
{
public:
  
  enum
  {
    RAMP_HSV      = 0,
    RAMP_VIRIDIS  = 1,
    RAMP_PLASMA   = 2,
    RAMP_MAGMA    = 3,
    RAMP_INFERNO  = 4,
    RAMP_CUSTOM   = 5
  };
  explicit UI(const VulkanMainContext& vmc);

  int selected_ramp = RAMP_HSV; 
  ImVec4 custom_start_color {1,1,0,1};
  ImVec4 custom_end_color {1,0,1,1};
  float custom_opacity_falloff = 1.0f;
  ImVec4 diff_color = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);
  bool diff_enabled = true;        
  ImVec4 intersect_color_common = ImVec4(1.0f, 0.5f, 0.0f, 1.0f);
  bool intersect_enabled_common = true;
  ImVec4 intersect_color_Aonly = ImVec4(1.0f, 0.0f, 0.0f, 0.3f); 
  bool intersect_enabled_Aonly = true;
  ImVec4 intersect_color_Bonly = ImVec4(0.0f, 0.0f, 1.0f, 0.3f);
  bool intersect_enabled_Bonly = true;
  ImVec4 union_color_Aonly = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
  bool union_enabled_Aonly = true;
  ImVec4 union_color_Bonly = ImVec4(0.0f, 0.0f, 1.0f, 1.0f);
  bool union_enabled_Bonly = true;
  ImVec4 union_color_common = ImVec4(1.0f, 0.0f, 1.0f, 1.0f);
  bool union_enabled_common = true;
  const Volume* gradient_volume = nullptr;
  void construct(VulkanCommandContext& vcc, const RenderPass& render_pass, uint32_t frames);
  void destruct();
  void draw(vk::CommandBuffer& cb, AppState& app_state);
  void set_transfer_function(TransferFunction* transfer_function);
  void set_volume(const Volume* volume);
  void set_persistence_pairs(const std::vector<PersistencePair>* pairs);
  void set_persistence_texture(ImTextureID tex);
  void set_on_pair_selected(const std::function<void(const PersistencePair&)>& callback);
  void set_on_range_applied(std::function<void(const std::vector<PersistencePair>&)> cb);
  void set_on_multi_selected(const std::function<void(const std::vector<PersistencePair>&)>& cb);
  void set_on_brush_selected(const std::function<void(const std::vector<PersistencePair>&)>& cb);
  void set_gradient_persistence_pairs(const std::vector<PersistencePair>* gp);
  void set_merge_tree(MergeTree* mt);
  void set_on_merge_mode_changed(const std::function<void(int)>& cb);
  void set_on_brush_selected_gradient(const std::function<void(const std::vector<std::pair<PersistencePair, float>>&, int)>& cb);
  void set_on_highlight_selected(const std::function<void(const std::vector<std::pair<PersistencePair,float>>&,int)>& cb);
  void clear_selection();
  void set_on_diff_selected(const std::function<void(const PersistencePair&, const PersistencePair&)>& cb);
  void set_on_intersect_selected(const std::function<void(const PersistencePair&, const PersistencePair&)>& cb);
  void set_on_union_selected(const std::function<void(const PersistencePair&, const PersistencePair&)>& cb);
  void mark_merge_tree_dirty();
  void set_on_custom_color_chosen(const std::function<void(const std::vector<PersistencePair>&, const ImVec4&)>& cb);
  void set_on_clear_custom_colors(const std::function<void()>& cb);
  void set_gradient_volume(const Volume* vol);
  void set_on_tf2d_selected(const std::function<void(const std::vector<std::pair<int,int>>&, const ImVec4&)>& cb);
  void set_on_reproject(const std::function<void()>& cb);
  void set_on_persistence_reprojected(const std::function<void(int featureIdx)> &user_cb);
  void set_on_persistence_multi_reprojected(const std::function<void(const std::vector<int>& featureIdxs)> &user_cb);
  void set_persistence_pairs(const std::vector<PersistencePair>* pairs,std::vector<std::vector<size_t>>&& voxel_indices);
  std::vector<std::pair<int,int>> persistence_bins;
  void set_on_evaluation(const std::function<void(float,float,float,float)>& cb);
  float last_J_arc = 0.0f;
  float last_J_box = 0.0f;
  float last_precision = 0.0f;
  float last_recall = 0.0f;
  bool  last_metrics_valid = false;
  bool pd_preview_active = false;
  std::vector<std::pair<int,int>> pd_preview_bins;
  std::vector<ImVec2> persistence_voxels;
  std::function<void(float J_arc, float J_box, float precision, float recall)> on_evaluation;
  std::vector<ImU32> persistence_bin_colors;
  
  const Volume* get_volume() const { return volume; }
  ImVec4 get_custom_start_color() const { return custom_start_color; }
  ImVec4 get_custom_end_color() const { return custom_end_color; }
  float get_custom_falloff() const { return custom_opacity_falloff; }
  int get_selected_ramp() const { return selected_ramp; }

private:
  const VulkanMainContext& vmc;
  vk::DescriptorPool imgui_pool;
  MergeTree* merge_tree = nullptr;
  TransferFunction* transfer_function = nullptr;
  const Volume* volume = nullptr;
  ImTextureID persistence_texture_ID = (ImTextureID)0;
  bool cache_dirty = true;
  bool show_dots = true;
  bool range_active = false;
  int max_points_to_show = 0;
  float normalization_factor = 255.0f;
  float diagram_zoom = 1.0f;
  float marker_size = 2.0f;
  float birth_range[2] = { 0.0f, 255.0f};
  float death_range[2] = { 0.0f, 255.0f};
  float persistence_range[2] = { 0.0f, 255.0f};
  float blink_timer = 0.0f;
  int selected_idx = -1; // no selection
  ImU32 selected_color = IM_COL32(255,0,255,255);
  ImVec2 brush_start;
  ImVec2 brush_end;
  float brush_outer_mult = 1.0f;
  float brush_inner_ratio = 0.7f;  
  bool use_gradient_pd = false;
  bool mt_dirty = true;
  bool use_highlight_opacity = false;
  float highlight_opacity = 1.0f;
  int selected_set_op = 0; // 0: diff, 1: intersect, 2: union
  ImVec4 selected_brush_color{1,0,0,1};

  bool tf2d_drag = false;
  ImVec2 tf2d_start;
  ImVec2 tf2d_end;
  bool region_defined = false;
  ImVec2 region_start;
  ImVec2 region_end;
  bool region_move = false;
  ImVec2 region_off;
  bool region_resize = false;
  int resize_corner = -1;
  float corner_r = 6.0f;
  bool brush_mode = false;
  float brush_radius_px    = 6.0f;
  bool brush_active = false;
  ImVec4 brush_color = ImVec4(0,1,1,1);
  std::vector<ImVec2> brush_points;
  int max_brush_hits = 1;
  ImVec4  rect_color = ImVec4(1,1,0,1);
  std::vector<std::vector<size_t>> persistence_voxel_indices;

  const std::vector<PersistencePair>* persistence_pairs = nullptr;
  std::vector<double> xs, ys;
  std::vector<float > pers;
  std::vector<ImVec2> dot_pos;
  std::vector<int> multi_selected_idxs;
  std::vector<ImU32> multi_selected_cols;
  const std::vector<PersistencePair>* gradient_pairs = nullptr;
  std::vector<std::pair<ImVec2,ImVec2>> mt_edges;
  std::vector<std::pair<ImVec2, uint32_t>> mt_nodes; 
  std::vector<std::pair<PersistencePair,float>> last_highlight_hits;
  std::vector<std::vector<int>> brush_clusters;
  std::vector<ImVec4> brush_cluster_colors;
  std::vector<ImU32> brush_cluster_outlines;
  std::vector<int> region_selected_idxs;
  std::vector<ImVec4> selected_custom_colors_per_point;
  std::vector<std::pair<int,int>> painted_bins;
  std::function<void(int)> on_merge_mode_changed;
  std::function<void(const std::vector<PersistencePair>&)> on_multi_selected;
  std::function<void(const std::vector<PersistencePair>&)> on_brush_selected;
  std::function<void(const std::vector<std::pair<PersistencePair, float>>& hits, int ramp)> on_brush_selected_gradient;
  std::function<void(const std::vector<std::pair<PersistencePair, float>>& hits, int ramp_index)> on_highlight_selected;
  std::function<void(const PersistencePair&, const PersistencePair&)> on_diff_selected;
  std::function<void(const PersistencePair&, const PersistencePair&)> on_intersect_selected;
  std::function<void(const PersistencePair&, const PersistencePair&)> on_union_selected;
  std::function<void(const PersistencePair&)> on_pair_selected;
  std::function<void(const std::vector<PersistencePair>&)> on_range_applied;
  std::function<void()> on_clear_custom_colors;
  std::function<void(const std::vector<PersistencePair>&, const ImVec4&)> on_color_chosen{};
  std::function<void(const std::vector<std::pair<int,int>>&, const ImVec4&)> on_tf2d_selected;
  std::function<void()> on_reproject;
  std::function<void(int)> on_persistence_reprojected;
  std::function<void(const std::vector<int>& featureIdxs)> on_persistence_multi_reprojected;
};
} // namespace ve