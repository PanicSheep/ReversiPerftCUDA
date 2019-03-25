#pragma once
#include <cassert>
#include <cstdint>
#include <string>

#if !(defined(_M_X64) || defined(__x86_64))
	#error "This code only works on x64!"
#endif

#if defined(_MSC_VER)
	#include <intrin.h>
	#pragma intrinsic(_BitScanForward64)
	#pragma intrinsic(_BitScanReverse64)
#elif defined(__GNUC__)
	#include <x86intrin.h>
#else
	#error compiler not supported
#endif

// Predefined macros hell:
//
// __GNUC__           Compiler is gcc, clang or Intel on Linux.
// __INTEL_COMPILER   Compiler is Intel.
// _MSC_VER           Compiler is MSVC or Intel on Windows.
// _WIN32             Building on Windows (any).
// _WIN64             Building on Windows 64 bit.
// _M_X64             Microsoft specific macro for 64 bit based machines.
// __x86_64           Defined by GNU C and Sun Studio for 64 bit based machines.

// #####################################
// ##    CPU specific optimizations   ##
// #####################################
//#define __core2__
//#define __corei7__
//#define __corei7_avx__
//#define __core_avx_i__
//#define __core_avx2__
//#define __bdver1__

//#define HAS_MMX
//#define HAS_SSE
//#define HAS_SSE2
//#define HAS_SSE3
//#define HAS_SSSE3
//#define HAS_SSE4_1
//#define HAS_SSE4_2
//#define HAS_SSE4a
//#define HAS_AVX
//#define HAS_AVX2
//#define HAS_BMI1
//#define HAS_BMI2
//#define HAS_ABM
//#define HAS_TBM
//#define HAS_POPCNT
//#define HAS_LZCNT
// #####################################

// CPUs instruction sets
#ifdef __core2__
	#define HAS_MMX
	#define HAS_SSE
	#define HAS_SSE2
	#define HAS_SSE3
	#define HAS_SSSE3
	#define HAS_SSE4_1
#endif
#ifdef __corei7__
	#define HAS_MMX
	#define HAS_SSE
	#define HAS_SSE2
	#define HAS_SSE3
	#define HAS_SSSE3
	#define HAS_SSE4_1
	#define HAS_SSE4_2
#endif
#if defined(__corei7_avx__) || defined(__core_avx_i__)
	#define HAS_MMX
	#define HAS_SSE
	#define HAS_SSE2
	#define HAS_SSE3
	#define HAS_SSSE3
	#define HAS_SSE4_1
	#define HAS_SSE4_2
#endif
#ifdef __core_avx2__
	#define HAS_MMX
	#define HAS_SSE
	#define HAS_SSE2
	#define HAS_SSE3
	#define HAS_SSSE3
	#define HAS_SSE4_1
	#define HAS_SSE4_2
	#define HAS_AVX
	#define HAS_AVX2
	#define HAS_BMI1
	#define HAS_BMI2
	#define HAS_POPCNT
	#define HAS_LZCNT
#endif
#ifdef __bdver1__
	#define HAS_MMX
	#define HAS_SSE
	#define HAS_SSE2
	#define HAS_SSE3
	#define HAS_SSSE3
	#define HAS_SSE4_1
	#define HAS_SSE4_2
	#define HAS_SSE4a
	#define HAS_AVX
	#define HAS_BMI1
	#define HAS_ABM
	#define HAS_TBM
#endif

// CPU instruction implications
#ifdef HAS_BMI1
	#define HAS_BEXTR	// Bit Field Extract
	#define HAS_BLSI	// Extract Lowest Set Isolated Bit  (x & -x)
	#define HAS_BLSMASK	// Get mask up to lowest set bit    (x ^ (x - 1))
	#define HAS_BLSR	// Reset lowest set bit             (x & (x - 1))
	#define HAS_LZCNT	// Leading Zero Count
	#define HAS_TZCNT	// Trailing Zero Count
#endif
#ifdef HAS_BMI2
	#define HAS_BZHI // Zero high bits starting with specified bit position
	#define HAS_PDEP // Parallel bits deposit
	#define HAS_PEXT // Parallel bits extract
#endif
#ifdef HAS_ABM
	#define HAS_POPCNT // Population count
	#define HAS_LZCNT  // Leading Zero Count
#endif
#ifdef HAS_TBM
	#define HAS_BEXTR	// Bit Field Extract
	#define HAS_BLCFILL	// Fill from lowest clear bit               ( x &  (x + 1))
	#define HAS_BLCI	// Isolate lowest clear bit                 ( x | ~(x + 1))
	#define HAS_BLCIC	// Isolate lowest clear bit and complement  (~x &  (x + 1))
	#define HAS_BLCMASK	// Mask from lowest clear bit               ( x ^  (x + 1))
	#define HAS_BLCS	// Set lowest clear bit                     ( x |  (x + 1))
	#define HAS_BLSFILL	// Fill from lowest set bit                 ( x |  (x - 1))
	#define HAS_BLSIC	// Isolate lowest set bit and complement    (~x |  (x - 1))
	#define HAS_T1MSKC	// Inverse mask from trailing ones          (~x |  (x + 1))
	#define HAS_TZMSK	// Mask from trailing zeros                 (~x &  (x - 1))
#endif

inline unsigned long BitScanLSB(const uint64_t mask) noexcept
{
	// OUT: mask == 0 results in undefined index
	assert(mask);
	#if defined(_MSC_VER)
		unsigned long index = 0;
		_BitScanForward64(&index, mask);
		return index;
	#elif defined(__GNUC__)
		return __builtin_ctzll(mask);
	#endif
}
	
inline unsigned long BitScanMSB(const uint64_t mask) noexcept
{
	assert(mask);
	#if defined(_MSC_VER)
		unsigned long index = 0;
		_BitScanReverse64(&index, mask);
		return index;
	#elif defined(__GNUC__)
		return __builtin_clzll(mask) ^ 63;
	#endif
}

inline uint64_t CountLeadingZeros(const uint64_t mask) noexcept
{
	// _lzcnt_u64(0) == 64
	// __builtin_clzll(0) is undefined
	assert(mask);
	#if defined(_MSC_VER)
		return _lzcnt_u64(mask);
	#elif defined(__GNUC__)
		return __builtin_clzll(mask);
	#endif
}

inline uint64_t CountTrailingZeros(const uint64_t mask) noexcept
{
	// _tzcnt_u64(0) is undefined
	// __builtin_ctzll(0) is undefined
	assert(mask);
	#if defined(_MSC_VER)
			return _tzcnt_u64(mask);
	#elif defined(__GNUC__)
			return __builtin_ctzll(mask);
	#endif
}

inline uint64_t GetLSB(const uint64_t b) noexcept
{
	#ifdef HAS_BLSI
		return _blsi_u64(b); 
	#else
		#pragma warning(disable : 4146)
		return b & -b;
	#endif
}

inline uint64_t GetMSB(const uint64_t b) noexcept
{ 
	return b != 0u ? 0x8000000000000000ui64 >> CountLeadingZeros(b) : 0;
}

inline void RemoveLSB(uint64_t & b) noexcept
{
	#ifdef HAS_BLSR
		b = _blsr_u64(b);
	#else
		b &= b - 1;
	#endif
}

inline void RemoveMSB(uint64_t & b) noexcept
{
	b ^= GetMSB(b);
}

template <typename T> inline uint64_t  MakeBit (                   const T pos ) noexcept { assert(pos < 64); return         1ui64 << pos; }
template <typename T> inline void       SetBit (      uint64_t& b, const T pos ) noexcept { assert(pos < 64);         b |=  (1ui64 << pos); }
template <typename T> inline void     ResetBit (      uint64_t& b, const T pos ) noexcept { assert(pos < 64);         b &= ~(1ui64 << pos); }
template <typename T> inline bool      TestBit (const uint64_t  b, const T pos ) noexcept { assert(pos < 64); return (b &   (1ui64 << pos)) != 0; }
template <typename T> inline bool      TestBits(const uint64_t  b, const T mask) noexcept { return (b & mask) == mask; }

inline uint64_t PopCount(uint64_t b) noexcept
{
	#ifdef HAS_POPCNT
		#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
			return _mm_popcnt_u64(b);
		#else
			return __builtin_popcountll(b);
		#endif
	#else
		b -= (b >> 1) & 0x5555555555555555ui64;
		b = ((b >> 2) & 0x3333333333333333ui64) + (b & 0x3333333333333333ui64);
		b = ((b >> 4) + b) & 0x0F0F0F0F0F0F0F0Fui64;
		return (b * 0x0101010101010101ui64) >> 56;
	#endif
}

inline uint64_t BExtr(const uint64_t src, const unsigned int start, unsigned int len) noexcept
{
	#if defined(HAS_BEXTR) || defined(HAS_TBM)
		return _bextr_u64(src, start, len);
	#else
		return (src >> start) & ((1ui64 << len) - 1);
	#endif
}

inline uint64_t BZHI(const uint64_t src, const uint32_t index) noexcept
{
	#ifdef HAS_BZHI
		return _bzhi_u64(src, index);
	#else
		return src & ((1ui64 << index) - 1);
	#endif
}

#ifdef HAS_BLCFILL
#else
#endif

#ifdef HAS_BLCI
#else
#endif

#ifdef HAS_BLCIC
#else
#endif

#ifdef HAS_BLCMASK
#else
#endif

#ifdef HAS_BLCS
#else
#endif

#ifdef HAS_BLSFILL
#else
#endif

#ifdef HAS_BLSIC
#else
#endif

#ifdef HAS_T1MSKC
#else
#endif

#ifdef HAS_TZMSK
#else
#endif

inline uint64_t PDep(uint64_t src, uint64_t mask) noexcept
{
	#ifdef HAS_PDEP
		return _pdep_u64(src, mask);
	#else
		uint64_t res = 0;
		for (uint64_t bb = 1; mask != 0u; bb += bb)
		{
			if ((src & bb) != 0u)
				res |= GetLSB(mask);
			RemoveLSB(mask);
		}
		return res;
	#endif
}

inline uint64_t PExt(uint64_t src, uint64_t mask) noexcept
{
	#ifdef HAS_PEXT
		return _pext_u64(src, mask);
	#else
		uint64_t res = 0;
		for (uint64_t bb = 1; mask != 0u; bb += bb)
		{
			if ((src & GetLSB(mask)) != 0u)
				res |= bb;
			RemoveLSB(mask);
		}
		return res;
	#endif
}

inline uint64_t BSwap(const uint64_t b) noexcept
{
	#if defined(_MSC_VER)
		return _byteswap_uint64(b);
	#elif defined(__GNUC__)
		return __builtin_bswap64(b);
	#endif
}


#if defined(_MSC_VER)
    #ifdef HAS_SSE2
        inline __m128i operator~(const __m128i& a) noexcept { return _mm_andnot_si128(a, _mm_set1_epi64x(0xFFFFFFFFFFFFFFFFui64)); }

        inline __m128i operator+(const __m128i& a, const __m128i& b) noexcept { return _mm_add_epi64(a, b); }
        inline __m128i operator-(const __m128i& a, const __m128i& b) noexcept { return _mm_sub_epi64(a, b); }
        inline __m128i operator&(const __m128i& a, const __m128i& b) noexcept { return _mm_and_si128(a, b); }
        inline __m128i operator|(const __m128i& a, const __m128i& b) noexcept { return _mm_or_si128(a, b); }
        inline __m128i operator^(const __m128i& a, const __m128i& b) noexcept { return _mm_xor_si128(a, b); }
        inline __m128i operator<<(const __m128i& a, const int b) noexcept { return _mm_slli_epi64(a, b); }
        inline __m128i operator>>(const __m128i& a, const int b) noexcept { return _mm_srli_epi64(a, b); }

        inline __m128i operator+=(__m128i& a, const __m128i& b) noexcept { return a = a + b; }
        inline __m128i operator-=(__m128i& a, const __m128i& b) noexcept { return a = a - b; }
        inline __m128i operator&=(__m128i& a, const __m128i& b) noexcept { return a = a & b; }
        inline __m128i operator|=(__m128i& a, const __m128i& b) noexcept { return a = a | b; }
        inline __m128i operator^=(__m128i& a, const __m128i& b) noexcept { return a = a ^ b; }
        inline __m128i operator<<=(__m128i& a, const int b) noexcept { return a = a << b; }
        inline __m128i operator>>=(__m128i& a, const int b) noexcept { return a = a >> b; }
    #endif
    #ifdef HAS_SSE4_1
        inline __m128i operator==(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpeq_epi64(a, b); }
    #endif
    #ifdef HAS_SSE4_2
        inline __m128i operator>(const __m128i& a, const __m128i& b) noexcept { return _mm_cmpgt_epi64(a, b); }
        inline __m128i operator<(const __m128i& a, const __m128i& b) noexcept { return b > a; }
        inline __m128i operator>=(const __m128i& a, const __m128i& b) noexcept { return ~(a < b); }
        inline __m128i operator<=(const __m128i& a, const __m128i& b) noexcept { return ~(a > b); }
    #endif
    #ifdef HAS_AVX2
		inline __m256i operator~(const __m256i& a) noexcept { return _mm256_xor_si256(a, _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFui64)); }

        inline __m256i operator+(const __m256i& a, const __m256i& b) noexcept { return _mm256_add_epi64(a, b); }
        inline __m256i operator-(const __m256i& a, const __m256i& b) noexcept { return _mm256_sub_epi64(a, b); }
        inline __m256i operator&(const __m256i& a, const __m256i& b) noexcept { return _mm256_and_si256(a, b); }
        inline __m256i operator|(const __m256i& a, const __m256i& b) noexcept { return _mm256_or_si256(a, b); }
        inline __m256i operator^(const __m256i& a, const __m256i& b) noexcept { return _mm256_xor_si256(a, b); }
        inline __m256i operator<<(const __m256i& a, const int b) noexcept { return _mm256_slli_epi64(a, b); }
        inline __m256i operator>>(const __m256i& a, const int b) noexcept { return _mm256_srli_epi64(a, b); }

        inline __m256i operator+=(__m256i& a, const __m256i& b) noexcept { return a = a + b; }
        inline __m256i operator-=(__m256i& a, const __m256i& b) noexcept { return a = a - b; }
        inline __m256i operator&=(__m256i& a, const __m256i& b) noexcept { return a = a & b; }
        inline __m256i operator|=(__m256i& a, const __m256i& b) noexcept { return a = a | b; }
        inline __m256i operator^=(__m256i& a, const __m256i& b) noexcept { return a = a ^ b; }
        inline __m256i operator<<=(__m256i& a, const int b) noexcept { return a = a << b; }
        inline __m256i operator>>=(__m256i& a, const int b) noexcept { return a = a >> b; }
    #endif
	#ifdef HAS_AVX512
		inline __m512i operator~(const __m512i& a) noexcept { return _mm512_xor_si512(a, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFFui64)); }

		inline __m512i operator+(const __m512i& a, const __m512i& b) noexcept { return _mm512_add_epi64(a, b); }
		inline __m512i operator-(const __m512i& a, const __m512i& b) noexcept { return _mm512_sub_epi64(a, b); }
		inline __m512i operator&(const __m512i& a, const __m512i& b) noexcept { return _mm512_and_si512(a, b); }
		inline __m512i operator|(const __m512i& a, const __m512i& b) noexcept { return _mm512_or_si512(a, b); }
		inline __m512i operator^(const __m512i& a, const __m512i& b) noexcept { return _mm512_xor_si512(a, b); }
		inline __m512i operator<<(const __m512i& a, const int b) noexcept { return _mm512_slli_epi64(a, b); }
		inline __m512i operator>>(const __m512i& a, const int b) noexcept { return _mm512_srli_epi64(a, b); }

		inline __m512i operator+=(__m512i& a, const __m512i& b) noexcept { return a = a + b; }
		inline __m512i operator-=(__m512i& a, const __m512i& b) noexcept { return a = a - b; }
		inline __m512i operator&=(__m512i& a, const __m512i& b) noexcept { return a = a & b; }
		inline __m512i operator|=(__m512i& a, const __m512i& b) noexcept { return a = a | b; }
		inline __m512i operator^=(__m512i& a, const __m512i& b) noexcept { return a = a ^ b; }
		inline __m512i operator<<=(__m512i& a, const int b) noexcept { return a = a << b; }
		inline __m512i operator>>=(__m512i& a, const int b) noexcept { return a = a >> b; }
	#endif
#endif


#ifdef _WIN32
	const std::string       FOLDER_SEPARATOR = "\\";
	const std::string WRONG_FOLDER_SEPARATOR = "/";
#else
	const std::string       FOLDER_SEPARATOR = "/";
	const std::string WRONG_FOLDER_SEPARATOR = "\\";
#endif
