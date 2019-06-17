#pragma once
#include "cuda_runtime.h"
#include "Chronosity.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cassert>

template<typename> class HostVector;

template <typename T>
class DeviceVector
{
	T * m_vec = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;
public:
	DeviceVector() = default;
	__host__ DeviceVector(const DeviceVector<T>& o) : m_vec(o.m_vec), m_size(o.m_size), m_capacity(0) {}
	//DeviceVector(const DeviceVector<T>& o) = delete;
	DeviceVector<T>& operator=(const DeviceVector<T>&) = delete;

	__host__            explicit DeviceVector(std::size_t capacity);
	__host__ __device__ DeviceVector(DeviceVector<T>&&) noexcept;
	__host__            DeviceVector<T>& operator=(const std::vector<T>&);
	__host__            DeviceVector<T>& operator=(const HostVector<T>&);
	__host__ __device__ DeviceVector<T>& operator=(DeviceVector<T>&&) noexcept;
	__host__            ~DeviceVector();

	__host__ DeviceVector<T> DeepCopy() const;
	__host__ DeviceVector<T> ShallowCopy();

	__host__ void assign(const std::vector<T>&);
	__host__ void assign(const HostVector<T>&, chronosity);
	__host__ void assign(const DeviceVector<T>&);

	__host__ void store(const std::vector<T>&);
	__host__ void store(const HostVector<T>&, chronosity);
	__host__ void store(const DeviceVector<T>&);

	__device__       T& operator[](std::size_t pos)       noexcept { return m_vec[pos]; }
	__device__ const T& operator[](std::size_t pos) const noexcept { return m_vec[pos]; }
	__device__       T& at(std::size_t pos)       noexcept(false);
	__device__ const T& at(std::size_t pos) const noexcept(false);
	__device__       T& front()       noexcept { return m_vec[0]; }
	__device__ const T& front() const noexcept { return m_vec[0]; }
	__device__       T& back()       noexcept { return m_vec[m_size - 1]; }
	__device__ const T& back() const noexcept { return m_vec[m_size - 1]; }
	__host__ __device__       T* data()       noexcept { return m_vec; }
	__host__ __device__ const T* data() const noexcept { return m_vec; }

	__host__ __device__       T*  begin()       noexcept { return m_vec; }
	__host__ __device__ const T*  begin() const noexcept { return m_vec; }
	__host__ __device__ const T* cbegin() const noexcept { return m_vec; }
	__host__ __device__       T*  end()       noexcept { return m_vec + m_size; }
	__host__ __device__ const T*  end() const noexcept { return m_vec + m_size; }
	__host__ __device__ const T* cend() const noexcept { return m_vec + m_size; }

	__host__ __device__ bool empty() const noexcept { return m_size == 0; }
	__host__ __device__ std::size_t size() const noexcept { return m_size; }
	__host__ __device__ std::size_t capacity() const noexcept { return m_capacity; }
	__host__            void reserve(std::size_t new_capacity);

	__host__ __device__ void clear() noexcept { m_size = 0; }
	__device__ void push_back(const T&);
	__device__ void push_back(T&&);
	__device__ void pop_back();
	__host__            void resize(std::size_t count);
	__host__ __device__ void swap(DeviceVector<T>&) noexcept;
};

template <typename T>
__host__ __device__ inline void swap(DeviceVector<T>& lhs, DeviceVector<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
{
	lhs.swap(rhs);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ DeviceVector<T>::DeviceVector(const std::size_t capacity) : m_capacity(capacity)
{
	cudaMalloc(&m_vec, capacity * sizeof(T));
}

template <typename T>
__host__ __device__ DeviceVector<T>::DeviceVector(DeviceVector<T>&& o) noexcept
{
	swap(o);
}

template <typename T>
__host__ DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& o)
{
	store(o);
	return *this;
}

template <typename T>
__host__ DeviceVector<T>& DeviceVector<T>::operator=(const HostVector<T>& o)
{
	store(o, syn);
	return *this;
}

template <typename T>
__host__ __device__ DeviceVector<T>& DeviceVector<T>::operator=(DeviceVector<T>&& o) noexcept
{
	swap(o);
	return *this;
}

template <typename T>
__host__ DeviceVector<T>::~DeviceVector()
{
	if (m_capacity)
		cudaFree(m_vec);
}

template <typename T>
__host__ void DeviceVector<T>::assign(const std::vector<T>& src)
{
	assert(m_capacity >= src.size());

	cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);

	m_size = src.size();
}

template <typename T>
__host__ void DeviceVector<T>::assign(const HostVector<T>& src, chronosity chrono)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (chrono == chronosity::syn)
		cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
	else
		cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);

	m_size = src.size();
}

template <typename T>
__host__ void DeviceVector<T>::assign(const DeviceVector<T>& src)
{
	assert(m_capacity >= src.size());

	cudaMemcpy(m_vec, src.m_vec, src.size() * sizeof(T), cudaMemcpyDeviceToDevice);

	m_size = src.size();
}

template <typename T>
__host__ void DeviceVector<T>::store(const std::vector<T>& src)
{
	if (m_capacity < src.size())
	{
		DeviceVector<T> new_vec{ src.size() };
		swap(new_vec);
	}
	assign(src);
}

template <typename T>
__host__ void DeviceVector<T>::store(const HostVector<T>& src, chronosity chrono)
{
	if (m_capacity < src.size())
	{
		DeviceVector<T> new_vec{ src.size() };
		swap(new_vec);
	}
	assign(src, chrono);
}

template <typename T>
__host__ void DeviceVector<T>::store(const DeviceVector<T>& src)
{
	if (m_capacity < src.size())
	{
		DeviceVector<T> new_vec{ src.size() };
		swap(new_vec);
	}
	assign(src);
}

template <typename T>
__device__ T & DeviceVector<T>::at(std::size_t pos)
{
	if (pos < size())
		return m_vec[pos];
	throw std::out_of_range{ "" };
}

template <typename T>
__device__ const T & DeviceVector<T>::at(std::size_t pos) const
{
	if (pos < size())
		return m_vec[pos];
	throw std::out_of_range{ "" };
}

template <typename T>
__host__ void DeviceVector<T>::reserve(const std::size_t new_capacity)
{
	if (new_capacity > m_capacity)
	{
		DeviceVector<T> new_vec{ new_capacity };
		new_vec.assign(*this);
		swap(new_vec);
	}
}

template <typename T>
__device__ void DeviceVector<T>::push_back(const T& value)
{
	if (m_size >= m_capacity)
		throw std::runtime_error{ "Not enough memory" };
	m_vec[m_size++] = value;
}

template <typename T>
__device__ void DeviceVector<T>::push_back(T&& value)
{
	if (m_size >= m_capacity)
		throw std::runtime_error{ "Not enough memory" };
	m_vec[m_size++] = std::move(value);
}

template <typename T>
__device__ void DeviceVector<T>::pop_back()
{
	m_size--;
}

template <typename T>
__host__ void DeviceVector<T>::resize(std::size_t count)
{
	reserve(count);
	m_size = count;
}

template <typename T>
__host__ __device__ void DeviceVector<T>::swap(DeviceVector<T>& o) noexcept
{
	::swap(m_vec, o.m_vec);
	::swap(m_size, o.m_size);
	::swap(m_capacity, o.m_capacity);
}
