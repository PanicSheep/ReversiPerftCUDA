#pragma once
#include "Chronosity.h"
#include "DeviceVector.cuh"
#include <cstdint>
#include <utility>
#include <vector>

template <class _Ty, class = void>
static constexpr bool is_iterator_v = false;

template <class T>
static constexpr bool is_iterator_v<T, std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;

// page-locked host memory
template <typename T>
class PinnedVector
{
	T* m_vec = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;

	void grow_if_needed() {
		if (m_size == m_capacity)
			reserve(std::max(m_capacity * 2, 1ULL));
	}
public:
	PinnedVector() = default;
	PinnedVector(PinnedVector<T>&& o) noexcept { swap(o); }
	PinnedVector(const PinnedVector<T>& o, chronosity c = syn) : PinnedVector(o.size()) { assign(o, c); }
	PinnedVector(const CudaVector<T>& o, chronosity c = syn) : PinnedVector(o.size()) { assign(o, c); }
	explicit PinnedVector(const std::vector<T>& o) : PinnedVector(o.size()) { assign(o); }
	explicit PinnedVector(std::size_t count) : m_size(count), m_capacity(count) { ErrorCheck([&]() { return cudaMallocHost(&m_vec, count * sizeof(T)); }); }
	~PinnedVector() { ErrorCheck([&]() { return cudaFreeHost(m_vec); }); }

	template <typename Iterator, std::enable_if_t<is_iterator_v<Iterator>, int> = 0>
	PinnedVector(Iterator begin, Iterator end) { store(begin, end); }

	PinnedVector<T>& operator=(PinnedVector<T>&& o) noexcept { swap(o); }
	PinnedVector<T>& operator=(const PinnedVector<T>& o) { store(o, syn); return *this; }
	PinnedVector<T>& operator=(const std::vector<T>& o) { store(o); return *this; }
	PinnedVector<T>& operator=(const CudaVector<T>& o) { store(o, syn); return *this; }


	// Assigns data. Requires sufficient capacity.
	void assign(const PinnedVector<T>&, chronosity);
	void assign(const CudaVector<T>&, chronosity, cudaStream_t = 0);
	void assign(const std::vector<T>&);

	// Stores data. Allocates memory if needed.
	template <typename Iterator, std::enable_if_t<is_iterator_v<Iterator>, int> = 0>
	void store(Iterator begin, Iterator end)
	{
		clear();
		while (begin != end) {
			push_back(*begin);
			++begin;
		}
	}
	void store(const PinnedVector<T>&, chronosity);
	void store(const CudaVector<T>&, chronosity, cudaStream_t = 0);
	void store(const std::vector<T>&);

	[[nodiscard]] std::vector<T> load() const { return { begin(), end() }; }

	[[nodiscard]] bool operator==(const PinnedVector<T>& o) const noexcept { return (m_size == o.size()) && std::equal(begin(), end(), o.begin(), o.end()); }
	[[nodiscard]] bool operator!=(const PinnedVector<T>& o) const noexcept { return !(*this == o); }

	[[nodiscard]]       T& operator[](std::size_t pos)       noexcept { return m_vec[pos]; }
	[[nodiscard]] const T& operator[](std::size_t pos) const noexcept { return m_vec[pos]; }
	[[nodiscard]]       T& at(std::size_t pos)       noexcept(false);
	[[nodiscard]] const T& at(std::size_t pos) const noexcept(false);
	[[nodiscard]]       T& front()       noexcept { return m_vec[0]; }
	[[nodiscard]] const T& front() const noexcept { return m_vec[0]; }
	[[nodiscard]]       T& back()       noexcept { return m_vec[m_size - 1]; }
	[[nodiscard]] const T& back() const noexcept { return m_vec[m_size - 1]; }
	[[nodiscard]]       T* data()       noexcept { return m_vec; }
	[[nodiscard]] const T* data() const noexcept { return m_vec; }
	[[nodiscard]]       T* begin()       noexcept { return m_vec; }
	[[nodiscard]] const T* begin() const noexcept { return m_vec; }
	[[nodiscard]] const T* cbegin() const noexcept { return m_vec; }
	[[nodiscard]]       T* end()       noexcept { return m_vec + m_size; }
	[[nodiscard]] const T* end() const noexcept { return m_vec + m_size; }
	[[nodiscard]] const T* cend() const noexcept { return m_vec + m_size; }

	[[nodiscard]] bool empty() const noexcept { return m_size == 0; }
	[[nodiscard]] std::size_t size() const noexcept { return m_size; }
	[[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }

	void clear() noexcept { m_size = 0; }
	void push_back(const T& value) { grow_if_needed(); m_vec[m_size++] = value; }
	void push_back(T&& value) { grow_if_needed(); m_vec[m_size++] = value; }
	void pop_back() { assert(m_size); m_size--; }
	void resize(std::size_t count) { reserve(count); m_size = count; }
	void reserve(std::size_t new_capacity);
	void swap(PinnedVector<T>& o) noexcept
	{
		std::swap(m_vec, o.m_vec);
		std::swap(m_size, o.m_size);
		std::swap(m_capacity, o.m_capacity);
	}
	void swap(PinnedVector<T>&& o) noexcept
	{
		std::swap(m_vec, o.m_vec);
		std::swap(m_size, o.m_size);
		std::swap(m_capacity, o.m_capacity);
	}
};

template <typename T>
inline void swap(PinnedVector<T>& l, PinnedVector<T>& r) noexcept { l.swap(r); }

template <typename T>
T& PinnedVector<T>::at(std::size_t pos) noexcept(false)
{
	if (pos >= m_size)
		throw std::out_of_range{ "Index out of range" };
	return m_vec[pos];
}

template <typename T>
const T& PinnedVector<T>::at(std::size_t pos) const noexcept(false)
{
	if (pos >= m_size)
		throw std::out_of_range{ "Index out of range" };
	return m_vec[pos];
}

template<typename T>
void PinnedVector<T>::assign(const PinnedVector<T>& src, chronosity c)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (c == chronosity::syn)
		ErrorCheck([&]() { return cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToHost); });
	else
		ErrorCheck([&]() { return cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToHost); });

	m_size = src.size();
}

template<typename T>
void PinnedVector<T>::assign(const CudaVector<T>& src, chronosity c, cudaStream_t stream)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (c == chronosity::syn)
		ErrorCheck([&]() { return cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost); });
	else
		ErrorCheck([&]() { return cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToHost, stream); });

	m_size = src.size();
}

template<typename T>
void PinnedVector<T>::assign(const std::vector<T>& src)
{
	assert(m_capacity >= src.size());

	std::copy(src.begin(), src.end(), m_vec);

	m_size = src.size();
}

template<typename T>
void PinnedVector<T>::store(const PinnedVector<T>& src, chronosity c)
{
	if (m_capacity < src.size())
		swap(PinnedVector<T>{ src.size() });
	assign(src, c);
}

template<typename T>
void PinnedVector<T>::store(const CudaVector<T>& src, chronosity c, cudaStream_t stream)
{
	if (m_capacity < src.size())
		swap(PinnedVector<T>{ src.size() });
	assign(src, c, stream);
}

template<typename T>
void PinnedVector<T>::store(const std::vector<T>& src)
{
	if (m_capacity < src.size())
		swap(PinnedVector<T>{ src.size() });
	assign(src);
}

template<typename T>
void PinnedVector<T>::reserve(std::size_t new_capacity)
{
	if (new_capacity > m_capacity)
	{
		PinnedVector<T> novum{ new_capacity };
		novum.assign(*this, syn);
		swap(novum);
	}
}
