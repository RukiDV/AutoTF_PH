#pragma once

#include "ve_log.hpp"
#include "vk/vulkan_main_context.hpp"
#include <ratio>

namespace ve
{
class DeviceTimer
{
public:
	enum TimerNames
	{
		RENDERING_ALL = 0,
		VOLUME = 1,
		UI = 2,
		TIMER_COUNT
	};

	DeviceTimer(const VulkanMainContext& vmc);
	void destruct();
	void reset(vk::CommandBuffer& cb, const std::vector<TimerNames>& timers);
	void start(vk::CommandBuffer& cb, TimerNames t, vk::PipelineStageFlagBits stage);
	void stop(vk::CommandBuffer& cb, TimerNames t, vk::PipelineStageFlagBits stage);

	template<class Precision = std::milli>
	double inline get_result_by_idx(uint32_t i)
	{
		VE_ASSERT(i < TIMER_COUNT, "Trying to access timer index that does not exist in TimerNames");
		return fetch_result<Precision>(i);
	}

	template<class Precision = std::milli>
	double inline get_result(TimerNames t)
	{
		return fetch_result<Precision>(t);
	}

private:
	template<class Precision>
	double inline fetch_result(uint32_t i)
	{
		// prevent repeated reading of the same timestamp
		if (result_fetched[i]) return -1.0;
		result_fetched[i] = true;
		std::array<uint64_t, 2> results;
		vk::Result result = vmc.logical_device.get().getQueryPoolResults(qp, i * 2, 2, results.size() * sizeof(uint64_t), results.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
		return (result == vk::Result::eSuccess) ? (double(timestamp_period * (results[1] - results[0])) / double(std::ratio_divide<std::nano, Precision>::den)) : -1.0;
	}

	const VulkanMainContext& vmc;
	vk::QueryPool qp;
	float timestamp_period;
	std::vector<bool> result_fetched;
};
} // namespace ve