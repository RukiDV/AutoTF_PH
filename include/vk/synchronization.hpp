#pragma once

#include "vk/common.hpp"

namespace ve
{
class Synchronization
{
public:
  enum SemaphoreNames
  {
    S_IMAGE_AVAILABLE = 0,
    S_RENDER_FINISHED = 1,
    S_HISTOGRAM_CLEAR_FINISHED = 2,
    SEMAPHORE_COUNT
  };

  enum FenceNames
  {
    F_RENDER_FINISHED = 0,
    F_COMPUTE_FINISHED = 1,
    F_COPY_FINISHED = 2,
    FENCE_COUNT
  };

  Synchronization(const vk::Device& logical_device);
  void destruct();
  const vk::Semaphore& get_semaphore(SemaphoreNames name) const;
  const vk::Fence& get_fence(FenceNames name) const;
  bool is_fence_finished(FenceNames name) const;
  void wait_for_fence(FenceNames name) const;
  void reset_fence(FenceNames name) const;

private:
  const vk::Device& device;
  std::vector<vk::Semaphore> semaphores;
  std::vector<vk::Fence> fences;
};
} // namespace ve
