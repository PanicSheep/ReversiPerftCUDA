#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cstdint>
#include <vector>

void Initialize();

class CPosition;

uint64_t perft_gpu(const std::vector<CPosition>&, int gpu_depth, int blocks, int threads_per_block);

#define HDI __host__ __device__ __inline__

//template <typename T>
//class unique_device_ptr
//{
//	T* p = nullptr;
//public:
//	HDI unique_device_ptr(T* p) : p(p) {}
//	HDI unique_device_ptr(std::size_t size) { cudaMalloc(&p, sizeof(T) * size); }
//
//	HDI unique_device_ptr(const std::vector<T>& o) : unique_device_ptr(o.size()) { cudaMemcpy( }
//
//	HDI unique_device_ptr(const unique_device_ptr&) = default;
//	HDI unique_device_ptr(unique_device_ptr&&) = default;
//	HDI ~unique_device_ptr() { cudaFree(p); }
//
//	HDI unique_device_ptr& operator=(const unique_device_ptr&) = default;
//	HDI unique_device_ptr& operator=(unique_device_ptr&&) = default;
//};
//
//template <class T, std::enable_if_t<std::is_array_v<T> && std::extent_v<T> == 0, int> = 0>
//HDI inline unique_device_ptr<T> make_unique_device(std::size_t size)
//{
//	typedef std::remove_extent_t<T> Elem;
//	return unique_device_ptr<T>(new Elem[size]());
//}

// Dynamic size on host
// Static size on device
//template <typename T>
//class DeviceArray
//{
//	T* device_ptr = nullptr;
//	std::size_t m_size;
//	int m_device_id;
//
//	void alloc() { cudaMalloc(&device_ptr, m_size * sizeof(T)); }
//	void free() { cudaFree(device_ptr); }
//public:
//	HDI MemoryGuard(const std::size_t size, int device_id = 0) : m_size(size), m_device_id(device_id)
//	{
//		cudaSetDevice(device_id);
//		alloc();
//	}
//	HDI DiviceArray(const DiviceArray&) = delete;
//	HDI DiviceArray(DiviceArray&&) = default;
//	HDI ~DiviceArray() { free(); }
//
//	HDI DiviceArray& operator=(const DiviceArray&) = delete;
//	HDI DiviceArray& operator=(DiviceArray&&) = default;
//};
//
//class Device