#pragma once
#include "cuda_runtime.h"
#include "Chronosity.h"
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cassert>

template<typename> class DeviceVector;

// page-locked host memory
template <typename T>
class HostVector
{
	T * m_vec = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;
public:
	HostVector() = default;
	__host__ explicit HostVector(std::size_t capacity);
	__host__ HostVector(const HostVector<T>&, chronosity = syn);
	__host__ HostVector(HostVector<T>&& o) noexcept;
	__host__ HostVector<T>& operator=(const std::vector<T>&);
	__host__ HostVector<T>& operator=(const HostVector<T>&);
	__host__ HostVector<T>& operator=(const DeviceVector<T>&);
	__host__ HostVector<T>& operator=(HostVector<T>&&) noexcept;
	__host__ ~HostVector();

	__host__ void assign(const std::vector<T>&);
	__host__ void assign(const HostVector<T>&, chronosity);
	__host__ void assign(const DeviceVector<T>&, chronosity);

	__host__ void store(const std::vector<T>&);
	__host__ void store(const HostVector<T>&, chronosity);
	__host__ void store(const DeviceVector<T>&, chronosity);

	__host__ std::vector<T> load() const;

	__host__       T& operator[](std::size_t pos)       noexcept { return m_vec[pos]; }
	__host__ const T& operator[](std::size_t pos) const noexcept { return m_vec[pos]; }
	__host__       T& at(std::size_t pos)       noexcept(false);
	__host__ const T& at(std::size_t pos) const noexcept(false);
	__host__       T& front()       noexcept { return m_vec[0]; }
	__host__ const T& front() const noexcept { return m_vec[0]; }
	__host__       T& back()       noexcept { return m_vec[m_size - 1]; }
	__host__ const T& back() const noexcept { return m_vec[m_size - 1]; }
	__host__       T* data()       noexcept { return m_vec; }
	__host__ const T* data() const noexcept { return m_vec; }

	__host__       T*  begin()       noexcept { return m_vec; }
	__host__ const T*  begin() const noexcept { return m_vec; }
	__host__ const T* cbegin() const noexcept { return m_vec; }
	__host__       T*  end()       noexcept { return m_vec + m_size; }
	__host__ const T*  end() const noexcept { return m_vec + m_size; }
	__host__ const T* cend() const noexcept { return m_vec + m_size; }

	__host__ bool empty() const noexcept { return m_size == 0; }
	__host__ std::size_t size() const noexcept { return m_size; }
	__host__ std::size_t capacity() const noexcept { return m_capacity; }
	__host__ void reserve(std::size_t new_capacity);

	__host__ void clear() noexcept { m_size = 0; }
	__host__ void push_back(const T&);
	__host__ void push_back(T&&);
	__host__ void pop_back();
	__host__ void resize(std::size_t count);
	__host__ void swap(HostVector<T>&) noexcept;
};

template <typename T>
__host__ inline void swap(HostVector<T>& lhs, HostVector<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
{
	lhs.swap(rhs);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ HostVector<T>::HostVector(const std::size_t capacity) : m_capacity(capacity)
{
	cudaMallocHost(&m_vec, capacity * sizeof(T));
}

template <typename T>
__host__ HostVector<T>::HostVector(const HostVector<T>& o, chronosity chrono) : HostVector(o.size())
{
	assign(o, chrono);
}

template <typename T>
__host__ HostVector<T>::HostVector(HostVector<T>&& o) noexcept
{
	swap(o);
}

template<typename T>
__host__ HostVector<T>& HostVector<T>::operator=(const std::vector<T>& o)
{
	store(o);
	return *this;
}

template<typename T>
__host__ HostVector<T>& HostVector<T>::operator=(const HostVector<T>& o)
{
	store(o, syn);
	return *this;
}

template<typename T>
__host__ HostVector<T>& HostVector<T>::operator=(const DeviceVector<T>& o)
{
	store(o, syn);
	return *this;
}

template<typename T>
__host__ HostVector<T>& HostVector<T>::operator=(HostVector<T>&& o) noexcept
{
	swap(o);
	return *this;
}

template<typename T>
__host__ HostVector<T>::~HostVector()
{
	cudaFreeHost(m_vec);
	m_size = 0;
	m_capacity = 0;
}

template<typename T>
__host__ void HostVector<T>::assign(const std::vector<T>& src)
{
	assert(m_capacity >= src.size()); 

	std::copy(src.begin(), src.end(), m_vec);

	m_size = src.size();
}

template<typename T>
__host__ void HostVector<T>::assign(const HostVector<T>& src, chronosity chrono)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (chrono == chronosity::syn)
		cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToHost);
	else
		cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToHost);
	
	m_size = src.size();
}

template<typename T>
__host__ void HostVector<T>::assign(const DeviceVector<T>& src, chronosity chrono)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (chrono == chronosity::syn)
		cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost);
	else
		cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost);

	m_size = src.size();
}

template<typename T>
__host__ void HostVector<T>::store(const std::vector<T>& src)
{
	if (m_capacity < src.size())
	{
		HostVector<T> new_vec{ src.size() };
		swap(new_vec);
	}
	assign(src);
}

template<typename T>
__host__ void HostVector<T>::store(const HostVector<T>& src, chronosity chrono)
{
	if (m_capacity < src.size())
	{
		HostVector<T> new_vec{ src.size() };
		swap(new_vec);
	}
	assign(src, chrono);
}

template<typename T>
__host__ void HostVector<T>::store(const DeviceVector<T>& src, chronosity chrono)
{
	if (m_capacity < src.size())
	{
		HostVector<T> new_vec{ src.size() };
		swap(new_vec);
	}
	assign(src, chrono);
}

template<typename T>
__host__ std::vector<T> HostVector<T>::load() const
{
	return { begin(), end() };
}

template<typename T>
__host__ T & HostVector<T>::at(std::size_t pos)
{
	if (pos < size())
		return m_vec[pos];
	throw std::out_of_range{ "" };
}

template<typename T>
__host__ const T & HostVector<T>::at(std::size_t pos) const
{
	if (pos < size())
		return m_vec[pos];
	throw std::out_of_range{ "" };
}

template<typename T>
__host__ void HostVector<T>::reserve(const std::size_t new_capacity)
{
	if (new_capacity > m_capacity)
	{
		HostVector<T> new_vec{ new_capacity };
		new_vec.assign(*this, syn);
		swap(new_vec);
	}
}

template<typename T>
__host__ void HostVector<T>::push_back(const T& value)
{
	if (m_size >= m_capacity)
		reserve(m_size * 2);
	m_vec[m_size++] = value;
}

template<typename T>
__host__ void HostVector<T>::push_back(T&& value)
{
	if (m_size >= m_capacity)
		reserve(m_size * 2);
	m_vec[m_size++] = std::move(value);
}

template<typename T>
__host__ void HostVector<T>::pop_back()
{
	m_size--;
}

template<typename T>
__host__ void HostVector<T>::resize(std::size_t count)
{
	reserve(count);
	m_size = count;
}

template<typename T>
__host__ void HostVector<T>::swap(HostVector<T>& o) noexcept
{
	std::swap(m_vec, o.m_vec);
	std::swap(m_size, o.m_size);
	std::swap(m_capacity, o.m_capacity);
}