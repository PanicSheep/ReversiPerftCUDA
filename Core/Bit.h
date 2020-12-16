#pragma once
#include "MoreTypes.h"

#ifdef __NVCC__
namespace std
{
	#ifdef __CUDA_ARCH__
	__device__ inline int countl_zero(uint64_t x) noexcept { return __clzll(x); }
	__device__ inline int countl_one(uint64_t x) noexcept { return countl_zero(~x); }
	__device__ inline int countr_zero(uint64_t x) noexcept { return __ffsll(x) - 1; }
	__device__ inline int countr_one(uint64_t x) noexcept { return countr_zero(~x); }
	__device__ inline int popcount(uint64_t x) noexcept { return __popcll(x); }
	#else
	__host__ inline int countl_zero(uint64_t x) noexcept { return static_cast<int>(_lzcnt_u64(x)); }
	__host__ inline int countl_one(uint64_t x) noexcept { return static_cast<int>(countl_zero(~x)); }
	__host__ inline int countr_zero(uint64_t x) noexcept { return static_cast<int>(_tzcnt_u64(x)); }
	__host__ inline int countr_one(uint64_t x) noexcept { return static_cast<int>(countr_zero(~x)); }
	__host__ inline int popcount(uint64_t x) noexcept { return static_cast<int>(_mm_popcnt_u64(x)); }
	#endif
}
#else
#include <bit>
#include <compare>
#endif

#ifdef __CUDA_ARCH__
	#define CUDA_CALLABLE __host__ __device__
#else
	#define CUDA_CALLABLE
#endif


[[nodiscard]]
CUDA_CALLABLE inline uint64_t GetLSB(uint64_t b) noexcept
{
	#pragma warning(suppress : 4146)
	return b & -b;
}

CUDA_CALLABLE inline void RemoveLSB(uint64_t& b) noexcept
{
	b &= b - 1;
}

[[nodiscard]]
CUDA_CALLABLE inline uint64_t BExtr(uint64_t src, uint start, uint len) noexcept
{
	#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
		return _bextr_u64(src, start, len);
	#else
		return (src >> start) & ((1ULL << len) - 1);
	#endif
}

[[nodiscard]]
inline uint64_t PDep(uint64_t src, uint64_t mask) noexcept
{
	return _pdep_u64(src, mask);
}

[[nodiscard]]
inline uint64_t PExt(uint64_t src, uint64_t mask) noexcept
{
	return _pext_u64(src, mask);
}

[[nodiscard]]
CUDA_CALLABLE inline uint64_t BSwap(uint64_t b) noexcept
{
	#if defined(__CUDA_ARCH__)
		return __byte_perm(b, 0, 0x0123);
	#elif defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}
