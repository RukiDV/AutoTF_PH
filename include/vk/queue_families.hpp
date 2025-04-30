#pragma once

#include <cstdint>
#include "common.hpp"
#include "util/named_bitfield.hpp"

enum class QueueFamilyFlags : uint32_t
{
	Graphics = (1 << 0),
	Compute = (1 << 1),
	Transfer = (1 << 2),
	Present = (1 << 3)
};

using Queues = NamedBitfield<QueueFamilyFlags>;
ENABLE_ENUM_OPERATORS(QueueFamilyFlags);

namespace ve
{
class QueueFamilies {
public:
  QueueFamilies() = default;
	void construct(vk::PhysicalDevice physical_device);
	void construct(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface);
	std::vector<uint32_t> get(Queues queues) const;
	uint32_t get(QueueFamilyFlags queue) const;

private:
  void get_queue_families(vk::PhysicalDevice& physical_device, const std::optional<vk::SurfaceKHR>& surface);

	struct Indices
	{
		int32_t graphics = -1;
		int32_t compute = -1;
		int32_t transfer = -1;
		int32_t present = -1;
	} indices;

	Queues graphics = QueueFamilyFlags::Graphics;
	Queues compute = QueueFamilyFlags::Compute;
	Queues transfer = QueueFamilyFlags::Transfer;
	Queues present = QueueFamilyFlags::Present;
};
} // namespace ve
