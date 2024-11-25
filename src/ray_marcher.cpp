#include "ray_marcher.hpp"
#include <vulkan/vulkan_handles.hpp>

namespace ve
{
Ray_Marcher::Ray_Marcher(const VulkanMainContext& vmc, Storage& storage) : vmc(vmc), storage(storage), clear_pipeline(vmc), pipeline(vmc), dsh(vmc, frames_in_flight)
{
  clear_storage_indices();
}

void Ray_Marcher::setup_storage(AppState& app_state, const Scene& scene)
{
  // set up ray marcher buffer
  std::vector<uint32_t> initial_ray_macher_data(app_state.get_render_extent().width * app_state.get_render_extent().height, 0);
  buffers[RAY_MARCHER_BUFFER_0] = storage.add_buffer("ray_marcher_output_0", initial_ray_macher_data, vk::BufferUsageFlagBits::eStorageBuffer, false, vmc.queue_family_indices.transfer, vmc.queue_family_indices.compute);
  buffers[RAY_MARCHER_BUFFER_1] = storage.add_buffer("ray_marcher_output_1", initial_ray_macher_data, vk::BufferUsageFlagBits::eStorageBuffer, false, vmc.queue_family_indices.transfer, vmc.queue_family_indices.compute);
  buffers[VOLUME_BUFFER] = storage.add_buffer("volume", scene.data, vk::BufferUsageFlagBits::eStorageBuffer, false, vmc.queue_family_indices.transfer, vmc.queue_family_indices.compute);
  std::vector<unsigned char> initial_image(app_state.get_render_extent().width * app_state.get_render_extent().height * 4, 0);
  images[RAY_MARCHER_IMAGE] = storage.add_image("ray_marcher_output_texture", initial_image.data(), app_state.get_render_extent().width, app_state.get_render_extent().height, false, 0, std::vector<uint32_t>{vmc.queue_family_indices.graphics, vmc.queue_family_indices.transfer, vmc.queue_family_indices.compute}, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eStorage);
}

void Ray_Marcher::construct(AppState& app_state, VulkanCommandContext& vcc, glm::uvec3 volume_resolution)
{
  std::cout << "Constructing histogram" << std::endl;
  for (uint32_t i : images) storage.get_image(i).transition_image_layout(vcc, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, vk::AccessFlagBits::eNone, vk::AccessFlagBits::eNone);
  create_descriptor_set();
  create_pipeline(app_state, volume_resolution);
  std::cout << "Successfully constructed histogram" << std::endl;
}

void Ray_Marcher::destruct()
{
  for (int32_t i : buffers)
  {
    if (i > -1) storage.destroy_buffer(i);
  }
  for (int32_t i : images)
  {
    if (i > -1) storage.destroy_image(i);
  }
  clear_storage_indices();
  clear_pipeline.destruct();
  pipeline.destruct();
  dsh.destruct();
}

void Ray_Marcher::compute(vk::CommandBuffer& cb, AppState& app_state, uint32_t read_only_buffer_idx)
{
  cb.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());
  cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline.get_layout(), 0, dsh.get_sets()[read_only_buffer_idx], {});
  cb.dispatch((app_state.get_render_extent().width + 31) / 32, (app_state.get_render_extent().height + 31) / 32, 1);
}

void Ray_Marcher::create_pipeline(const AppState& app_state, glm::uvec3 volume_resolution)
{
  std::array<vk::SpecializationMapEntry, 5> spec_entries;
  spec_entries[0] = vk::SpecializationMapEntry(0, 0, sizeof(uint32_t));
  spec_entries[1] = vk::SpecializationMapEntry(1, sizeof(uint32_t), sizeof(uint32_t));
  spec_entries[2] = vk::SpecializationMapEntry(2, sizeof(uint32_t) * 2, sizeof(uint32_t));
  spec_entries[3] = vk::SpecializationMapEntry(3, sizeof(uint32_t) * 3, sizeof(uint32_t));
  spec_entries[4] = vk::SpecializationMapEntry(4, sizeof(uint32_t) * 4, sizeof(uint32_t));
  std::array<uint32_t, 5> spec_entries_data{app_state.get_render_extent().width, app_state.get_render_extent().height, volume_resolution.x, volume_resolution.y, volume_resolution.z};
  vk::SpecializationInfo spec_info(spec_entries.size(), spec_entries.data(), sizeof(uint32_t) * spec_entries_data.size(), spec_entries_data.data());
  ShaderInfo ray_marcher_shader_info = ShaderInfo{"ray_marcher.comp", vk::ShaderStageFlagBits::eCompute, spec_info};
  pipeline.construct(dsh.get_layouts()[0], ray_marcher_shader_info, 0);
}

void Ray_Marcher::create_descriptor_set()
{
  dsh.add_binding(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
  dsh.add_binding(2, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute);
  dsh.add_binding(3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
  dsh.add_binding(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
  for (uint32_t i = 0; i < frames_in_flight; ++i)
  {
    // compute histogram from the read only spotlight_compute_buffer to the other histogram buffer
    // this ensures that the read_only_buffer_idx is valid for the spotlight_compute_buffer and the histogram_buffer
    dsh.add_descriptor(i, 0, storage.get_buffer_by_name("volume"));
    dsh.add_descriptor(i, 2, storage.get_image_by_name("ray_marcher_output_texture"));
    dsh.add_descriptor(i, 3, storage.get_buffer_by_name("ray_marcher_output_" + std::to_string(i)));
    dsh.add_descriptor(i, 4, storage.get_buffer_by_name("ray_marcher_output_" + std::to_string(1 - i)));
  }
  dsh.construct();
}

void Ray_Marcher::clear_storage_indices()
{
  for (uint32_t i = 0; i < BUFFER_COUNT; i++) buffers[i] = -1;
}
} // namespace ve
