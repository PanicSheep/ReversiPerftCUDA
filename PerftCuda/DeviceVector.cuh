#pragma once
#include "Chronosity.h"
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cassert>

#ifdef __CUDA_ARCH__
	#define DeviceCode
#else
	#define HostCode
#endif

template <typename>
class PinnedVector;

// Vector in cuda memory
template <typename T>
class CudaVector
{
	T* m_vec = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;
public:
	CudaVector() = default;
	__host__ CudaVector(CudaVector<T>&& o) noexcept { swap(o); }
	__host__ CudaVector(const CudaVector<T>& o, chronosity c = syn) : CudaVector(o.size()) { assign(o, c); }
	__host__ CudaVector(const PinnedVector<T>& o, chronosity c = syn) : CudaVector(o.size()) { assign(o, c); }
	__host__ explicit CudaVector(const std::vector<T>& o) : CudaVector(o.size()) { assign(o); }
	__host__ explicit CudaVector(std::size_t count) : m_size(count), m_capacity(count) { ErrorCheck([&](){ return cudaMalloc(&m_vec, count * sizeof(T)); }); }
	__host__ ~CudaVector() { ErrorCheck([&](){ return cudaFree(m_vec); }); }

	__host__ CudaVector<T>& operator=(CudaVector<T>&& o) noexcept { swap(o); }
	__host__ CudaVector<T>& operator=(const CudaVector<T>& o) { store(o, syn); return *this; }
	__host__ CudaVector<T>& operator=(const PinnedVector<T>& o) { store(o, syn); return *this; }
	__host__ CudaVector<T>& operator=(const std::vector<T>& o) { store(o); return *this; }


	bool operator==(const CudaVector<T>&) const noexcept = delete; // Too expensive!
	bool operator!=(const CudaVector<T>&) const noexcept = delete; // Too expensive!

	// Assigns data. Requires sufficient capacity.
	__host__ void assign(const PinnedVector<T>&, chronosity, cudaStream_t = 0);
	__host__ void assign(const CudaVector<T>&, chronosity);
	__host__ void assign(const std::vector<T>&);

	// Stores data. Allocates memory if needed.
	__host__ void store(const PinnedVector<T>&, chronosity, cudaStream_t = 0);
	__host__ void store(const CudaVector<T>&, chronosity);
	__host__ void store(const std::vector<T>&);

	__host__ std::vector<T> load() const;

	__host__ [[nodiscard]]       T* data()       noexcept { return m_vec; }
	__host__ [[nodiscard]] const T* data() const noexcept { return m_vec; }
	__host__ [[nodiscard]]       T* begin()       noexcept { return m_vec; }
	__host__ [[nodiscard]] const T* begin() const noexcept { return m_vec; }
	__host__ [[nodiscard]] const T* cbegin() const noexcept { return m_vec; }
	__host__ [[nodiscard]]       T* end()       noexcept { return m_vec + m_size; }
	__host__ [[nodiscard]] const T* end() const noexcept { return m_vec + m_size; }
	__host__ [[nodiscard]] const T* cend() const noexcept { return m_vec + m_size; }

	__host__ [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
	__host__ [[nodiscard]] std::size_t size() const noexcept { return m_size; }
	__host__ [[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }
	__host__ void reserve(std::size_t new_capacity, chronosity, cudaStream_t = 0);

	__host__ void clear() noexcept { m_size = 0; }
	__host__ void resize(std::size_t count, chronosity c, cudaStream_t stream = 0) { reserve(count, c, stream); m_size = count; }
	__host__ void swap(CudaVector<T>& o) noexcept
	{
		std::swap(m_vec, o.m_vec);
		std::swap(m_size, o.m_size);
		std::swap(m_capacity, o.m_capacity);
	}
	__host__ void swap(CudaVector<T>&& o) noexcept
	{
		std::swap(m_vec, o.m_vec);
		std::swap(m_size, o.m_size);
		std::swap(m_capacity, o.m_capacity);
	}
};

template <typename T>
__host__ inline void swap(CudaVector<T>& l, CudaVector<T>& r) noexcept { l.swap(r); }

// View on a vector in cuda memory
template <typename T>
class CudaVector_view
{
	T* m_vec = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;
public:
	CudaVector_view(CudaVector<T>& o) noexcept : m_vec(o.data()), m_size(o.size()), m_capacity(o.capacity()) {}
	CudaVector_view(const CudaVector_view<T>&) = default;
	CudaVector_view(CudaVector_view<T>&&) noexcept = default;
	CudaVector_view<T>& operator=(const CudaVector_view<T>&) = default;
	CudaVector_view<T>& operator=(CudaVector_view<T>&&) noexcept = default;
	~CudaVector_view() = default;

	bool operator==(const CudaVector_view<T>&) const noexcept = delete; // Too expensive!
	bool operator!=(const CudaVector_view<T>&) const noexcept = delete; // Too expensive!

	__device__ [[nodiscard]]       T& operator[](std::size_t pos)       noexcept { return m_vec[pos]; }
	__device__ [[nodiscard]] const T& operator[](std::size_t pos) const noexcept { return m_vec[pos]; }
	__device__ [[nodiscard]]       T& at(std::size_t pos)       noexcept(false);
	__device__ [[nodiscard]] const T& at(std::size_t pos) const noexcept(false);
	__device__ [[nodiscard]]       T& front()       noexcept { return m_vec[0]; }
	__device__ [[nodiscard]] const T& front() const noexcept { return m_vec[0]; }
	__device__ [[nodiscard]]       T& back()       noexcept { return m_vec[m_size - 1]; }
	__device__ [[nodiscard]] const T& back() const noexcept { return m_vec[m_size - 1]; }
	__device__ [[nodiscard]]       T* data()       noexcept { return m_vec; }
	__device__ [[nodiscard]] const T* data() const noexcept { return m_vec; }
	__device__ [[nodiscard]]       T* begin()       noexcept { return m_vec; }
	__device__ [[nodiscard]] const T* begin() const noexcept { return m_vec; }
	__device__ [[nodiscard]] const T* cbegin() const noexcept { return m_vec; }
	__device__ [[nodiscard]]       T* end()       noexcept { return m_vec + m_size; }
	__device__ [[nodiscard]] const T* end() const noexcept { return m_vec + m_size; }
	__device__ [[nodiscard]] const T* cend() const noexcept { return m_vec + m_size; }

	__device__ [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
	__device__ [[nodiscard]] std::size_t size() const noexcept { return m_size; }
	__device__ [[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }

	__device__ void clear() noexcept { m_size = 0; }
	__device__ void push_back(const T& value) noexcept(false);
	__device__ void push_back(T&& value) noexcept(false);
	__device__ void pop_back() { m_size--; }
	__device__ void swap(CudaVector_view<T>& o) noexcept
	{
		std::swap(m_vec, o.m_vec);
		std::swap(m_size, o.m_size);
		std::swap(m_capacity, o.m_capacity);
	}
	__device__ void swap(CudaVector_view<T>&& o) noexcept
	{
		std::swap(m_vec, o.m_vec);
		std::swap(m_size, o.m_size);
		std::swap(m_capacity, o.m_capacity);
	}
};

template <typename T>
__device__ inline void swap(CudaVector_view<T>& l, CudaVector_view<T>& r) noexcept { l.swap(r); }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void CudaVector<T>::assign(const PinnedVector<T>& src, chronosity c, cudaStream_t stream)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (c == chronosity::syn)
		ErrorCheck([&]() { return cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice); });
	else
		ErrorCheck([&]() { return cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream); });

	m_size = src.size();
}

template <typename T>
void CudaVector<T>::assign(const CudaVector<T>& src, chronosity c)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (c == chronosity::syn)
		ErrorCheck([&]() { return cudaMemcpy(m_vec, src.m_vec, src.size() * sizeof(T), cudaMemcpyDeviceToDevice); });
	else
		ErrorCheck([&]() { return cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice); });

	m_size = src.size();
}
template <typename T>
void CudaVector<T>::assign(const std::vector<T>& src)
{
	assert(m_capacity >= src.size());

	ErrorCheck([&]() { return cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice); });

	m_size = src.size();
}

template <typename T>
void CudaVector<T>::store(const PinnedVector<T>& src, chronosity c, cudaStream_t stream)
{
	if (m_capacity < src.size())
		swap(CudaVector<T>{ src.size() });
	assign(src, c, stream);
}

template <typename T>
void CudaVector<T>::store(const CudaVector<T>& src, chronosity c)
{
	if (m_capacity < src.size())
		swap(CudaVector<T>{ src.size() });
	assign(src, c);
}

template <typename T>
void CudaVector<T>::store(const std::vector<T>& src)
{
	if (m_capacity < src.size())
		swap(CudaVector<T>{ src.size() });
	assign(src);
}

template<typename T>
std::vector<T> CudaVector<T>::load() const
{
	std::vector<T> ret(m_size);
	ErrorCheck([&]() { return cudaMemcpy(ret.data(), m_vec, m_size * sizeof(T), cudaMemcpyDeviceToHost); });
	return ret;
}

template <typename T>
void CudaVector<T>::reserve(const std::size_t new_capacity, chronosity c, cudaStream_t stream)
{
	if (new_capacity > m_capacity)
	{
		CudaVector<T> novum{ new_capacity };
		novum.assign(*this, c, stream);
		swap(novum);
	}
}


template <typename T>
__device__ T& CudaVector_view<T>::at(std::size_t pos) noexcept(false)
{
	if (pos >= m_size)
		throw std::out_of_range{ "Index out of range" };
	return m_vec[pos];
}

template <typename T>
__device__ const T& CudaVector_view<T>::at(std::size_t pos) const noexcept(false)
{
	if (pos >= m_size)
		throw std::out_of_range{ "Index out of range" };
	return m_vec[pos];
}

template <typename T>
__device__ void CudaVector_view<T>::push_back(const T& value) noexcept(false)
{
	if (m_size >= m_capacity)
		throw std::runtime_error{ "Capacity is exhausted." };
	m_vec[m_size++] = value;
}

template <typename T>
__device__ void CudaVector_view<T>::push_back(T&& value) noexcept(false)
{
	if (m_size >= m_capacity)
		throw std::runtime_error{ "Capacity is exhausted." };
	m_vec[m_size++] = std::move(value);
}
