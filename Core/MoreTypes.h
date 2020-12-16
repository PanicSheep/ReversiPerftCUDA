#pragma once
#include <cstdint>
#include <type_traits>

// Predefined macros:
// __GNUC__           Compiler is gcc.
// __clang__          Compiler is clang.
// __INTEL_COMPILER   Compiler is Intel.
// _MSC_VER           Compiler is Microsoft Visual Studio.
// _M_X64             Microsoft specific macro when targeting 64 bit based machines.
// __x86_64           Defined by GNU C and Sun Studio when targeting 64 bit based machines.

// TODO: int's: gather, maskload, maskstore, permute, shuffle, hadd

#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(__GNUC__)
	#include <x86intrin.h>
#else
	#error compiler not supported!
#endif

#if !(defined(_M_X64) || defined(__x86_64))
	#error This code requires x64!
#endif

using uint = unsigned int;

using int8 = signed char;
using int16 = short;
using int32 = int;
using int64 = long long;

using uint8 = unsigned char;
using uint16 = unsigned short;
using uint32 = unsigned int;
using uint64 = unsigned long long;

constexpr int8 operator""_i8(uint64 v) { return int8(v); }
constexpr int16 operator""_i16(uint64 v) { return int16(v); }
constexpr int32 operator""_i32(uint64 v) { return int32(v); }
constexpr int64 operator""_i64(uint64 v) { return int64(v); }

constexpr uint8 operator""_ui8(uint64 v) { return uint8(v); }
constexpr uint16 operator""_ui16(uint64 v) { return uint16(v); }
constexpr uint32 operator""_ui32(uint64 v) { return uint32(v); }
constexpr uint64 operator""_ui64(uint64 v) { return uint64(v); }

#ifdef __AVX2__
template <class intN, uint M>
class intNxM;

using int8x16 = intNxM<int8, 16>;
using int8x32 = intNxM<int8, 32>;

using int16x8 = intNxM<int16, 8>;
using int16x16 = intNxM<int16, 16>;

using int32x4 = intNxM<int32, 4>;
using int32x8 = intNxM<int32, 8>;

using int64x2 = intNxM<int64, 2>;
using int64x4 = intNxM<int64, 4>;


namespace detail
{
	template <uint index> [[nodiscard]] inline int64 extract_epi64(__m128i reg) noexcept { static_assert(index <  2); _mm_extract_epi64(reg, index); }
	template <uint index> [[nodiscard]] inline int64 extract_epi64(__m256i reg) noexcept { static_assert(index <  4); _mm256_extract_epi64(reg, index); }
	template <uint index> [[nodiscard]] inline int32 extract_epi32(__m128i reg) noexcept { static_assert(index <  4); _mm_extract_epi32(reg, index); }
	template <uint index> [[nodiscard]] inline int32 extract_epi32(__m256i reg) noexcept { static_assert(index <  8); _mm256_extract_epi32(reg, index); }
	template <uint index> [[nodiscard]] inline int16 extract_epi16(__m128i reg) noexcept { static_assert(index <  8); _mm_extract_epi16(reg, index); }
	template <uint index> [[nodiscard]] inline int16 extract_epi16(__m256i reg) noexcept { static_assert(index < 16); _mm256_extract_epi16(reg, index); }
	template <uint index> [[nodiscard]] inline int8  extract_epi8 (__m128i reg) noexcept { static_assert(index < 16); _mm_extract_epi8(reg, index); }
	template <uint index> [[nodiscard]] inline int8  extract_epi8 (__m256i reg) noexcept { static_assert(index < 32); _mm256_extract_epi8(reg, index); }

	template <uint index> [[nodiscard]] inline __m128i insert_epi(__m128i reg, int64 v) noexcept { static_assert(index <  2); return _mm_insert_epi64(reg, v, index); }
	template <uint index> [[nodiscard]] inline __m256i insert_epi(__m256i reg, int64 v) noexcept { static_assert(index <  4); return _mm256_insert_epi64(reg, v, index); }
	template <uint index> [[nodiscard]] inline __m128i insert_epi(__m128i reg, int32 v) noexcept { static_assert(index <  4); return _mm_insert_epi32(reg, v, index); }
	template <uint index> [[nodiscard]] inline __m256i insert_epi(__m256i reg, int32 v) noexcept { static_assert(index <  8); return _mm256_insert_epi32(reg, v, index); }
	template <uint index> [[nodiscard]] inline __m128i insert_epi(__m128i reg, int16 v) noexcept { static_assert(index <  8); return _mm_insert_epi16(reg, v, index); }
	template <uint index> [[nodiscard]] inline __m256i insert_epi(__m256i reg, int16 v) noexcept { static_assert(index < 16); return _mm256_insert_epi16(reg, v, index); }
	template <uint index> [[nodiscard]] inline __m128i insert_epi(__m128i reg, int8  v) noexcept { static_assert(index < 16); return _mm_insert_epi8(reg, v, index); }
	template <uint index> [[nodiscard]] inline __m256i insert_epi(__m256i reg, int8  v) noexcept { static_assert(index < 32); return _mm256_insert_epi8(reg, v, index); }

	[[nodiscard]] inline __m128i and_si(__m128i l, __m128i r) noexcept { return _mm_and_si128(l, r); }
	[[nodiscard]] inline __m256i and_si(__m256i l, __m256i r) noexcept { return _mm256_and_si256(l, r); }
	[[nodiscard]] inline __m128i or_si(__m128i l, __m128i r) noexcept { return _mm_or_si128(l, r); }
	[[nodiscard]] inline __m256i or_si(__m256i l, __m256i r) noexcept { return _mm256_or_si256(l, r); }
	[[nodiscard]] inline __m128i xor_si(__m128i l, __m128i r) noexcept { return _mm_xor_si128(l, r); }
	[[nodiscard]] inline __m256i xor_si(__m256i l, __m256i r) noexcept { return _mm256_xor_si256(l, r); }
	[[nodiscard]] inline __m128i andnot_si(__m128i l, __m128i r) noexcept { return _mm_andnot_si128(l, r); }
	[[nodiscard]] inline __m256i andnot_si(__m256i l, __m256i r) noexcept { return _mm256_andnot_si256(l, r); }
};

template <class intN, uint M>
class intNxM
{
	constexpr static bool is_128_bit = sizeof(intN) * 8 * M == 128;
	constexpr static bool is_256_bit = sizeof(intN) * 8 * M == 256;

	static_assert(is_128_bit || is_256_bit);
	using Register = std::conditional_t<is_128_bit, __m128i, __m256i>;

	Register reg{0};
public:
	intNxM() noexcept = default;
	intNxM(Register v) noexcept : reg(v) {}
	explicit intNxM(intN) noexcept;
	explicit intNxM(std::nullptr_t) noexcept = delete;
	explicit intNxM(const intN* p) noexcept : reg(_mm256_load_si256(reinterpret_cast<const Register*>(p))) {}

	template <typename = std::enable_if_t<is_128_bit>>
	intNxM(int64 e1, int64 e0) noexcept : reg(_mm_set_epi64x(e1, e0)) {}

	template <typename = std::enable_if_t<is_128_bit>>
	intNxM(int32 e3, int32 e2, int32 e1, int32 e0) noexcept : reg(_mm_set_epi32(e3, e2, e1, e0)) {}

	template <typename = std::enable_if_t<is_128_bit>>
	intNxM(int16 e7, int16 e6, int16 e5, int16 e4, int16 e3, int16 e2, int16 e1, int16 e0) noexcept
		: reg(_mm_set_epi16(e7, e6, e5, e4, e3, e2, e1, e0))
	{}

	template <typename = std::enable_if_t<is_128_bit>>
	intNxM(int8 e15, int8 e14, int8 e13, int8 e12, int8 e11, int8 e10, int8 e9, int8 e8,
		   int8 e7, int8 e6, int8 e5, int8 e4, int8 e3, int8 e2, int8 e1, int8 e0) noexcept
		: reg(_mm_set_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0))
	{}

	template <typename = std::enable_if_t<is_256_bit>>
	intNxM(int64 e3, int64 e2, int64 e1, int64 e0) noexcept : reg(_mm256_set_epi64x(e3, e2, e1, e0)) {}

	template <typename = std::enable_if_t<is_256_bit>>
	intNxM(int32 e7, int32 e6, int32 e5, int32 e4, int32 e3, int32 e2, int32 e1, int32 e0) noexcept
		: reg(_mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0))
	{}

	template <typename = std::enable_if_t<is_256_bit>>
	intNxM(int16 e15, int16 e14, int16 e13, int16 e12, int16 e11, int16 e10, int16 e9, int16 e8,
		   int16 e7, int16 e6, int16 e5, int16 e4, int16 e3, int16 e2, int16 e1, int16 e0) noexcept
		: reg(_mm256_set_epi16(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0))
	{}

	template <typename = std::enable_if_t<is_256_bit>>
	intNxM(int8 e31, int8 e30, int8 e29, int8 e28, int8 e27, int8 e26, int8 e25, int8 e24,
		   int8 e23, int8 e22, int8 e21, int8 e20, int8 e19, int8 e18, int8 e17, int8 e16,
		   int8 e15, int8 e14, int8 e13, int8 e12, int8 e11, int8 e10, int8 e9, int8 e8,
		   int8 e7, int8 e6, int8 e5, int8 e4, int8 e3, int8 e2, int8 e1, int8 e0) noexcept
		: reg(_mm256_set_epi8(e31, e30, e29, e28, e27, e26, e25, e24, e23, e22, e21, e20, e19, e18, e17, e16,
							  e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0))
	{}


	[[nodiscard]] operator Register() const noexcept { return reg; }

	template <uint index> [[nodiscard]] intN  get() const noexcept;
	template <uint index> [[nodiscard]] int8  get_int8 () const noexcept { return detail::extract_epi8 <index>(reg); }
	template <uint index> [[nodiscard]] int16 get_int16() const noexcept { return detail::extract_epi16<index>(reg); }
	template <uint index> [[nodiscard]] int32 get_int32() const noexcept { return detail::extract_epi32<index>(reg); }
	template <uint index> [[nodiscard]] int64 get_int64() const noexcept { return detail::extract_epi64<index>(reg); }

	template <uint index> [[nodiscard]] intNxM<intN, M/2> get_int128() const noexcept {
		static_assert(is_256_bit && index < 2);
		return _mm256_extracti128_si256(reg, index);
	}

	template <uint index> void set(intN) noexcept;
	template <uint index> void set_int8 (intN v) noexcept { reg = detail::insert_epi<index>(reg, v); }
	template <uint index> void set_int16(intN v) noexcept { reg = detail::insert_epi<index>(reg, v); }
	template <uint index> void set_int32(intN v) noexcept { reg = detail::insert_epi<index>(reg, v); }
	template <uint index> void set_int64(intN v) noexcept { reg = detail::insert_epi<index>(reg, v); }

	template <uint index> void set_int128(intNxM<intN, M/2> v) const noexcept {
		static_assert(is_256_bit && index < 2);
		reg = _mm256_inserti128_si256(reg, v, index);
	}

	[[nodiscard]] bool operator==(intNxM o) const noexcept { return reduce_and(cmpeq(reg, o)); }
	[[nodiscard]] bool operator!=(intNxM o) const noexcept { return !(*this == o); }

	[[nodiscard]] intNxM operator~() const noexcept { return andnot(reg, intNxM{-1}); }
	[[nodiscard]] intNxM operator-() const noexcept { return intNxM{} - reg; }

	[[nodiscard]] intNxM operator&(intNxM v) const noexcept { return detail::and_si(reg, v); }
	[[nodiscard]] intNxM operator|(intNxM v) const noexcept { return detail::or_si(reg, v); }
	[[nodiscard]] intNxM operator^(intNxM v) const noexcept { return detail::xor_si(reg, v); }
	[[nodiscard]] friend intNxM andnot(intNxM l, intNxM r) noexcept { return detail::andnot_si(l, r); }

	intNxM& operator+=(intNxM v) noexcept { reg = *this + v; return *this; }
	intNxM& operator-=(intNxM v) noexcept { reg = *this - v; return *this; }
	intNxM& operator&=(intNxM v) noexcept { reg = *this & v; return *this; }
	intNxM& operator|=(intNxM v) noexcept { reg = *this | v; return *this; }
	intNxM& operator^=(intNxM v) noexcept { reg = *this ^ v; return *this; }
	intNxM& operator<<=(intNxM v) noexcept { reg = *this << v; return *this; }
	intNxM& operator>>=(intNxM v) noexcept { reg = *this >> v; return *this; }
	intNxM& operator<<=(int v) noexcept { reg = *this << v; return *this; }
	intNxM& operator>>=(int v) noexcept { reg = *this >> v; return *this; }

	[[nodiscard]] intNxM operator+(intNxM) const noexcept;
	[[nodiscard]] intNxM operator-(intNxM) const noexcept;
	[[nodiscard]] intNxM operator/(intNxM) const noexcept;
	[[nodiscard]] intNxM operator%(intNxM) const noexcept;
	[[nodiscard]] intNxM operator<<(intNxM) const noexcept;
	[[nodiscard]] intNxM operator>>(intNxM) const noexcept;
	[[nodiscard]] intNxM operator<<(int) const noexcept;
	[[nodiscard]] intNxM operator>>(int) const noexcept;
};

template <> inline intNxM<int64, 2>::intNxM(int64 v) noexcept : reg(_mm_set1_epi64x(v)) {}
template <> inline intNxM<int32, 4>::intNxM(int32 v) noexcept : reg(_mm_set1_epi32 (v)) {}
template <> inline intNxM<int16, 8>::intNxM(int16 v) noexcept : reg(_mm_set1_epi16 (v)) {}
template <> inline intNxM<int8 ,16>::intNxM(int8  v) noexcept : reg(_mm_set1_epi8  (v)) {}
template <> inline intNxM<int64, 4>::intNxM(int64 v) noexcept : reg(_mm256_set1_epi64x(v)) {}
template <> inline intNxM<int32, 8>::intNxM(int32 v) noexcept : reg(_mm256_set1_epi32 (v)) {}
template <> inline intNxM<int16,16>::intNxM(int16 v) noexcept : reg(_mm256_set1_epi16 (v)) {}
template <> inline intNxM<int8 ,32>::intNxM(int8  v) noexcept : reg(_mm256_set1_epi8  (v)) {}

template <> template <uint index> inline int64 intNxM<int64, 2>::get() const noexcept { static_assert(index <  2); return _mm_extract_epi64(reg, index); }
template <> template <uint index> inline int32 intNxM<int32, 4>::get() const noexcept { static_assert(index <  4); return _mm_extract_epi32(reg, index); }
template <> template <uint index> inline int16 intNxM<int16, 8>::get() const noexcept { static_assert(index <  8); return _mm_extract_epi16(reg, index); }
template <> template <uint index> inline int8  intNxM<int8 ,16>::get() const noexcept { static_assert(index < 16); return _mm_extract_epi8 (reg, index); }
template <> template <uint index> inline int64 intNxM<int64, 4>::get() const noexcept { static_assert(index <  4); return _mm256_extract_epi64(reg, index); }
template <> template <uint index> inline int32 intNxM<int32, 8>::get() const noexcept { static_assert(index <  8); return _mm256_extract_epi32(reg, index); }
template <> template <uint index> inline int16 intNxM<int16,16>::get() const noexcept { static_assert(index < 16); return _mm256_extract_epi16(reg, index); }
template <> template <uint index> inline int8  intNxM<int8 ,32>::get() const noexcept { static_assert(index < 32); return _mm256_extract_epi8 (reg, index); }

template <> template <uint index> inline void intNxM<int64, 2>::set(int64 v) noexcept { static_assert(index <  2); return _mm_insert_epi64(reg, v, index); }
template <> template <uint index> inline void intNxM<int32, 4>::set(int32 v) noexcept { static_assert(index <  4); return _mm_insert_epi32(reg, v, index); }
template <> template <uint index> inline void intNxM<int16, 8>::set(int16 v) noexcept { static_assert(index <  8); return _mm_insert_epi16(reg, v, index); }
template <> template <uint index> inline void intNxM<int8 ,16>::set(int8  v) noexcept { static_assert(index < 16); return _mm_insert_epi8 (reg, v, index); }
template <> template <uint index> inline void intNxM<int64, 4>::set(int64 v) noexcept { static_assert(index <  4); return _mm256_insert_epi64(reg, v, index); }
template <> template <uint index> inline void intNxM<int32, 8>::set(int32 v) noexcept { static_assert(index <  8); return _mm256_insert_epi32(reg, v, index); }
template <> template <uint index> inline void intNxM<int16,16>::set(int16 v) noexcept { static_assert(index < 16); return _mm256_insert_epi16(reg, v, index); }
template <> template <uint index> inline void intNxM<int8 ,32>::set(int8  v) noexcept { static_assert(index < 32); return _mm256_insert_epi8 (reg, v, index); }

inline intNxM<int64,2> intNxM<int64,2>::operator+(intNxM<int64,2> v) const noexcept { return _mm_add_epi64(reg, v); }
inline intNxM<int64,2> intNxM<int64,2>::operator-(intNxM<int64,2> v) const noexcept { return _mm_sub_epi64(reg, v); }
inline intNxM<int64,2> intNxM<int64,2>::operator/(intNxM<int64,2> v) const noexcept { return _mm_div_epi64(reg, v); }
inline intNxM<int64,2> intNxM<int64,2>::operator%(intNxM<int64,2> v) const noexcept { return _mm_rem_epi64(reg, v); }
inline intNxM<int64,2> intNxM<int64,2>::operator<<(intNxM<int64,2> v) const noexcept { return _mm_sllv_epi64(reg, v); }
inline intNxM<int64,2> intNxM<int64,2>::operator>>(intNxM<int64,2> v) const noexcept { return _mm_srlv_epi64(reg, v); }
inline intNxM<int64,2> intNxM<int64,2>::operator<<(int v) const noexcept { return _mm_slli_epi64(reg, v); }
inline intNxM<int64,2> intNxM<int64,2>::operator>>(int v) const noexcept { return _mm_srli_epi64(reg, v); }

inline intNxM<int32,4> intNxM<int32,4>::operator+(intNxM<int32,4> v) const noexcept { return _mm_add_epi32(reg, v); }
inline intNxM<int32,4> intNxM<int32,4>::operator-(intNxM<int32,4> v) const noexcept { return _mm_sub_epi32(reg, v); }
inline intNxM<int32,4> intNxM<int32,4>::operator/(intNxM<int32,4> v) const noexcept { return _mm_div_epi32(reg, v); }
inline intNxM<int32,4> intNxM<int32,4>::operator%(intNxM<int32,4> v) const noexcept { return _mm_rem_epi32(reg, v); }
inline intNxM<int32,4> intNxM<int32,4>::operator<<(intNxM<int32,4> v) const noexcept { return _mm_sllv_epi32(reg, v); }
inline intNxM<int32,4> intNxM<int32,4>::operator>>(intNxM<int32,4> v) const noexcept { return _mm_srlv_epi32(reg, v); }
inline intNxM<int32,4> intNxM<int32,4>::operator<<(int v) const noexcept { return _mm_slli_epi32(reg, v); }
inline intNxM<int32,4> intNxM<int32,4>::operator>>(int v) const noexcept { return _mm_srli_epi32(reg, v); }

inline intNxM<int16,8> intNxM<int16,8>::operator+(intNxM<int16,8> v) const noexcept { return _mm_add_epi16(reg, v); }
inline intNxM<int16,8> intNxM<int16,8>::operator-(intNxM<int16,8> v) const noexcept { return _mm_sub_epi16(reg, v); }
inline intNxM<int16,8> intNxM<int16,8>::operator/(intNxM<int16,8> v) const noexcept { return _mm_div_epi16(reg, v); }
inline intNxM<int16,8> intNxM<int16,8>::operator%(intNxM<int16,8> v) const noexcept { return _mm_rem_epi16(reg, v); }
inline intNxM<int16,8> intNxM<int16,8>::operator<<(int v) const noexcept { return _mm_slli_epi16(reg, v); }
inline intNxM<int16,8> intNxM<int16,8>::operator>>(int v) const noexcept { return _mm_srli_epi16(reg, v); }
//inline intNxM<int16,8> intNxM<int16,8>::operator<<(intNxM<int16,8> v) const noexcept { return _mm_sllv_epi16(reg, v); } // Instruction requires AVX512VL and AVX512BW!
//inline intNxM<int16,8> intNxM<int16,8>::operator>>(intNxM<int16,8> v) const noexcept { return _mm_srlv_epi16(reg, v); } // Instruction requires AVX512VL and AVX512BW!

inline intNxM<int8,16> intNxM<int8,16>::operator+(intNxM<int8,16> v) const noexcept { return _mm_add_epi8(reg, v); }
inline intNxM<int8,16> intNxM<int8,16>::operator-(intNxM<int8,16> v) const noexcept { return _mm_sub_epi8(reg, v); }
inline intNxM<int8,16> intNxM<int8,16>::operator/(intNxM<int8,16> v) const noexcept { return _mm_div_epi8(reg, v); }
inline intNxM<int8,16> intNxM<int8,16>::operator%(intNxM<int8,16> v) const noexcept { return _mm_rem_epi8(reg, v); }
//inline intNxM<int8,16> intNxM<int8,16>::operator<<(intNxM<int8,16> v) const noexcept { return _mm_sllv_epi8(reg, v); } // Instruction does not exist!
//inline intNxM<int8,16> intNxM<int8,16>::operator>>(intNxM<int8,16> v) const noexcept { return _mm_srlv_epi8(reg, v); } // Instruction does not exist!
//inline intNxM<int8,16> intNxM<int8,16>::operator<<(int v) const noexcept { return _mm_slli_epi8(reg, v); } // Instruction does not exist!
//inline intNxM<int8,16> intNxM<int8,16>::operator>>(int v) const noexcept { return _mm_srli_epi8(reg, v); } // Instruction does not exist!

inline intNxM<int64,4> intNxM<int64,4>::operator+(intNxM<int64,4> v) const noexcept { return _mm256_add_epi64(reg, v); }
inline intNxM<int64,4> intNxM<int64,4>::operator-(intNxM<int64,4> v) const noexcept { return _mm256_sub_epi64(reg, v); }
inline intNxM<int64,4> intNxM<int64,4>::operator/(intNxM<int64,4> v) const noexcept { return _mm256_div_epi64(reg, v); }
inline intNxM<int64,4> intNxM<int64,4>::operator%(intNxM<int64,4> v) const noexcept { return _mm256_rem_epi64(reg, v); }
inline intNxM<int64,4> intNxM<int64,4>::operator<<(intNxM<int64,4> v) const noexcept { return _mm256_sllv_epi64(reg, v); }
inline intNxM<int64,4> intNxM<int64,4>::operator>>(intNxM<int64,4> v) const noexcept { return _mm256_srlv_epi64(reg, v); }
inline intNxM<int64,4> intNxM<int64,4>::operator<<(int v) const noexcept { return _mm256_slli_epi64(reg, v); }
inline intNxM<int64,4> intNxM<int64,4>::operator>>(int v) const noexcept { return _mm256_srli_epi64(reg, v); }

inline intNxM<int32,8> intNxM<int32,8>::operator+(intNxM<int32,8> v) const noexcept { return _mm256_add_epi32(reg, v); }
inline intNxM<int32,8> intNxM<int32,8>::operator-(intNxM<int32,8> v) const noexcept { return _mm256_sub_epi32(reg, v); }
inline intNxM<int32,8> intNxM<int32,8>::operator/(intNxM<int32,8> v) const noexcept { return _mm256_div_epi32(reg, v); }
inline intNxM<int32,8> intNxM<int32,8>::operator%(intNxM<int32,8> v) const noexcept { return _mm256_rem_epi32(reg, v); }
inline intNxM<int32,8> intNxM<int32,8>::operator<<(intNxM<int32,8> v) const noexcept { return _mm256_sllv_epi32(reg, v); }
inline intNxM<int32,8> intNxM<int32,8>::operator>>(intNxM<int32,8> v) const noexcept { return _mm256_srlv_epi32(reg, v); }
inline intNxM<int32,8> intNxM<int32,8>::operator<<(int v) const noexcept { return _mm256_slli_epi32(reg, v); }
inline intNxM<int32,8> intNxM<int32,8>::operator>>(int v) const noexcept { return _mm256_srli_epi32(reg, v); }

inline intNxM<int16,16> intNxM<int16,16>::operator+(intNxM<int16,16> v) const noexcept { return _mm256_add_epi16(reg, v); }
inline intNxM<int16,16> intNxM<int16,16>::operator-(intNxM<int16,16> v) const noexcept { return _mm256_sub_epi16(reg, v); }
inline intNxM<int16,16> intNxM<int16,16>::operator/(intNxM<int16,16> v) const noexcept { return _mm256_div_epi16(reg, v); }
inline intNxM<int16,16> intNxM<int16,16>::operator%(intNxM<int16,16> v) const noexcept { return _mm256_rem_epi16(reg, v); }
inline intNxM<int16,16> intNxM<int16,16>::operator<<(int v) const noexcept { return _mm256_slli_epi16(reg, v); }
inline intNxM<int16,16> intNxM<int16,16>::operator>>(int v) const noexcept { return _mm256_srli_epi16(reg, v); }
//inline intNxM<int16,16> intNxM<int16,16>::operator<<(intNxM<int16,16> v) const noexcept { return _mm256_sllv_epi16(reg, v); } // Instruction requires AVX512VL and AVX512BW!
//inline intNxM<int16,16> intNxM<int16,16>::operator>>(intNxM<int16,16> v) const noexcept { return _mm256_srlv_epi16(reg, v); } // Instruction requires AVX512VL and AVX512BW!

inline intNxM<int8,32> intNxM<int8,32>::operator+(intNxM<int8,32> v) const noexcept { return _mm256_add_epi8(reg, v); }
inline intNxM<int8,32> intNxM<int8,32>::operator-(intNxM<int8,32> v) const noexcept { return _mm256_sub_epi8(reg, v); }
inline intNxM<int8,32> intNxM<int8,32>::operator/(intNxM<int8,32> v) const noexcept { return _mm256_div_epi8(reg, v); }
inline intNxM<int8,32> intNxM<int8,32>::operator%(intNxM<int8,32> v) const noexcept { return _mm256_rem_epi8(reg, v); }
//inline intNxM<int8,32> intNxM<int8,32>::operator<<(intNxM<int8,32> v) const noexcept { return _mm256_sllv_epi8(reg, v); } // Instruction does not exist!
//inline intNxM<int8,32> intNxM<int8,32>::operator>>(intNxM<int8,32> v) const noexcept { return _mm256_srlv_epi8(reg, v); } // Instruction does not exist!
//inline intNxM<int8,32> intNxM<int8,32>::operator<<(int v) const noexcept { return _mm256_slli_epi8(reg, v); } // Instruction does not exist!
//inline intNxM<int8,32> intNxM<int8,32>::operator>>(int v) const noexcept { return _mm256_srli_epi8(reg, v); } // Instruction does not exist!

// returns a==b ? -1 : 0
[[nodiscard]] inline intNxM<int64, 2> cmpeq(intNxM<int64, 2> a, intNxM<int64, 2> b) noexcept { return _mm_cmpeq_epi64(a, b); }
[[nodiscard]] inline intNxM<int32, 4> cmpeq(intNxM<int32, 4> a, intNxM<int32, 4> b) noexcept { return _mm_cmpeq_epi32(a, b); }
[[nodiscard]] inline intNxM<int16, 8> cmpeq(intNxM<int16, 8> a, intNxM<int16, 8> b) noexcept { return _mm_cmpeq_epi16(a, b); }
[[nodiscard]] inline intNxM<int8 ,16> cmpeq(intNxM<int8 ,16> a, intNxM<int8 ,16> b) noexcept { return _mm_cmpeq_epi8 (a, b); }
[[nodiscard]] inline intNxM<int64, 4> cmpeq(intNxM<int64, 4> a, intNxM<int64, 4> b) noexcept { return _mm256_cmpeq_epi64(a, b); }
[[nodiscard]] inline intNxM<int32, 8> cmpeq(intNxM<int32, 8> a, intNxM<int32, 8> b) noexcept { return _mm256_cmpeq_epi32(a, b); }
[[nodiscard]] inline intNxM<int16,16> cmpeq(intNxM<int16,16> a, intNxM<int16,16> b) noexcept { return _mm256_cmpeq_epi16(a, b); }
[[nodiscard]] inline intNxM<int8 ,32> cmpeq(intNxM<int8 ,32> a, intNxM<int8 ,32> b) noexcept { return _mm256_cmpeq_epi8 (a, b); }

// returns a>b ? -1 : 0
[[nodiscard]] inline intNxM<int64, 2> cmpgt(intNxM<int64, 2> a, intNxM<int64, 2> b) noexcept { return _mm_cmpgt_epi64(a, b); }
[[nodiscard]] inline intNxM<int32, 4> cmpgt(intNxM<int32, 4> a, intNxM<int32, 4> b) noexcept { return _mm_cmpgt_epi32(a, b); }
[[nodiscard]] inline intNxM<int16, 8> cmpgt(intNxM<int16, 8> a, intNxM<int16, 8> b) noexcept { return _mm_cmpgt_epi16(a, b); }
[[nodiscard]] inline intNxM<int8 ,16> cmpgt(intNxM<int8 ,16> a, intNxM<int8 ,16> b) noexcept { return _mm_cmpgt_epi8 (a, b); }
[[nodiscard]] inline intNxM<int64, 4> cmpgt(intNxM<int64, 4> a, intNxM<int64, 4> b) noexcept { return _mm256_cmpgt_epi64(a, b); }
[[nodiscard]] inline intNxM<int32, 8> cmpgt(intNxM<int32, 8> a, intNxM<int32, 8> b) noexcept { return _mm256_cmpgt_epi32(a, b); }
[[nodiscard]] inline intNxM<int16,16> cmpgt(intNxM<int16,16> a, intNxM<int16,16> b) noexcept { return _mm256_cmpgt_epi16(a, b); }
[[nodiscard]] inline intNxM<int8 ,32> cmpgt(intNxM<int8 ,32> a, intNxM<int8 ,32> b) noexcept { return _mm256_cmpgt_epi8 (a, b); }

[[nodiscard]] inline intNxM<int32, 4> abs(intNxM<int32, 4> v) noexcept { return _mm_abs_epi32(v); }
[[nodiscard]] inline intNxM<int16, 8> abs(intNxM<int16, 8> v) noexcept { return _mm_abs_epi16(v); }
[[nodiscard]] inline intNxM<int8 ,16> abs(intNxM<int8 ,16> v) noexcept { return _mm_abs_epi8 (v); }
[[nodiscard]] inline intNxM<int64, 4> abs(intNxM<int64, 4> v) noexcept { return _mm256_abs_epi64(v); }
[[nodiscard]] inline intNxM<int32, 8> abs(intNxM<int32, 8> v) noexcept { return _mm256_abs_epi32(v); }
[[nodiscard]] inline intNxM<int16,16> abs(intNxM<int16,16> v) noexcept { return _mm256_abs_epi16(v); }
[[nodiscard]] inline intNxM<int8 ,32> abs(intNxM<int8 ,32> v) noexcept { return _mm256_abs_epi8 (v); }
//[[nodiscard]] inline intNxM<int64, 2> abs(intNxM<int64, 2> v) noexcept { return _mm_abs_epi64(v); } // Instruction requires AVX512VL and AVX512F!


// unpackhi({a,_}, {c,_}) -> {c,a}
[[nodiscard]] inline intNxM<int64, 2> unpackhi(intNxM<int64, 2> x, intNxM<int64, 2> y) noexcept { return _mm_unpackhi_epi64(x, y); }

// unpacklo({_,b}, {_,d}) -> {d,b}
[[nodiscard]] inline intNxM<int64, 2> unpacklo(intNxM<int64, 2> x, intNxM<int64, 2> y) noexcept { return _mm_unpacklo_epi64(x, y); }


// unpackhi({a,b,_,_}, {e,f,_,_}) -> {e,a,f,b}
[[nodiscard]] inline intNxM<int32, 4> unpackhi(intNxM<int32, 4> x, intNxM<int32, 4> y) noexcept { return _mm_unpackhi_epi32(x, y); }

// unpacklo({_,_,c,d}, {_,_,g,h}) -> {g,c,h,d}
[[nodiscard]] inline intNxM<int32, 4> unpacklo(intNxM<int32, 4> x, intNxM<int32, 4> y) noexcept { return _mm_unpacklo_epi32(x, y); }


// unpackhi({a,b,c,d,_,_,_,_}, {i,j,k,l,_,_,_,_}) -> {i,a,j,b,k,c,l,d}
[[nodiscard]] inline intNxM<int16, 8> unpackhi(intNxM<int16, 8> x, intNxM<int16, 8> y) noexcept { return _mm_unpackhi_epi16(x, y); }

// unpacklo({_,_,_,_,e,f,g,h}, {_,_,_,_,m,n,o,p})) -> {m,e,n,f,o,g,p,h}
[[nodiscard]] inline intNxM<int16, 8> unpacklo(intNxM<int16, 8> x, intNxM<int16, 8> y) noexcept { return _mm_unpacklo_epi16(x, y); }


// unpackhi({a,b,c,d,e,f,g,h,_,_,_,_,_,_,_,_}, {a2,b2,c2,d2,e2,f2,g2,h2,_,_,_,_,_,_,_,_}) -> {a2,a,b2,b,c2,c,d2,d,e2,e,f2,f,g2,g,h2,h}
[[nodiscard]] inline intNxM<int8, 16> unpackhi(intNxM<int8, 16> x, intNxM<int8, 16> y) noexcept { return _mm_unpackhi_epi8(x, y); }

// unpacklo({_,_,_,_,_,_,_,_,i,j,k,l,m,n,o,p}, {_,_,_,_,_,_,_,_,i2,j2,k2,l2,m2,n2,o2,p2})) -> {i2,i,j2,j,k2,k,l2,l,m2,m,n2,n,o2,o,p2,p}
[[nodiscard]] inline intNxM<int8, 16> unpacklo(intNxM<int8, 16> x, intNxM<int8, 16> y) noexcept { return _mm_unpacklo_epi8(x, y); }


// unpackhi({a,_,c,_}, {e,_,g,_}) -> {e,a,g,c}
[[nodiscard]] inline intNxM<int64, 4> unpackhi(intNxM<int64, 4> x, intNxM<int64, 4> y) noexcept { return _mm256_unpackhi_epi64(x, y); }

// unpacklo({_,b,_,d}, {_,f,_,h}) -> {f,b,h,d}
[[nodiscard]] inline intNxM<int64, 4> unpacklo(intNxM<int64, 4> x, intNxM<int64, 4> y) noexcept { return _mm256_unpacklo_epi64(x, y); }


// unpackhi({a,b,_,_,e,f,_,_}, {i,j,_,_,m,n,_,_}) -> {i,a,j,b,m,e,n,f}
[[nodiscard]] inline intNxM<int32, 8> unpackhi(intNxM<int32, 8> x, intNxM<int32, 8> y) noexcept { return _mm256_unpackhi_epi32(x, y); }

// unpacklo({_,_,c,d,_,_,g,h}, {_,_,k,l,_,_,o,p})) -> {k,c,l,d,o,g,p,h}
[[nodiscard]] inline intNxM<int32, 8> unpacklo(intNxM<int32, 8> x, intNxM<int32, 8> y) noexcept { return _mm256_unpacklo_epi32(x, y); }


// unpackhi({a,b,c,d,_,_,_,_,i,j,k,l,_,_,_,_}, {a2,b2,c2,d2,_,_,_,_,i2,j2,k2,l2,_,_,_,_}) -> {a2,a,b2,b,c2,c,d2,d,i2,i,j2,j,k2,k,l2,l}
[[nodiscard]] inline intNxM<int16, 16> unpackhi(intNxM<int16, 16> x, intNxM<int16, 16> y) noexcept { return _mm256_unpackhi_epi16(x, y); }

// unpacklo({_,_,_,_,e,f,g,h,_,_,_,_,m,n,o,p}, {_,_,_,_,e2,f2,g2,h2,_,_,_,_,m2,n2,o2,p2})) -> {e2,e,f2,f,g2,g,h2,h,m2,m,n2,n,o2,o,p2,p}
[[nodiscard]] inline intNxM<int16, 16> unpacklo(intNxM<int16, 16> x, intNxM<int16, 16> y) noexcept { return _mm256_unpacklo_epi16(x, y); }


// unpackhi({a1,b1,c1,d1,e1,f1,g1,h1,_,_,_,_,_,_,_,_,i1,j1,k1,l1,m1,n1,o1,p1,_,_,_,_,_,_,_,_},{a2,b2,c2,d2,e2,f2,g2,h2,_,_,_,_,_,_,_,_,i2,j2,k2,l2,m2,n2,o2,p2,_,_,_,_,_,_,_,_})
//   -> {a2,a1,b2,b1,c2,c1,d2,d1,e2,e1,f2,f1,g2,g1,h2,h1,i2,i1,j2,j1,k2,k1,l2,l1,m2,m1,n2,n1,o2,o1,p2,p1}
[[nodiscard]] inline intNxM<int8, 32> unpackhi(intNxM<int8, 32> x, intNxM<int8, 32> y) noexcept { return _mm256_unpackhi_epi8(x, y); }

// unpacklo({_,_,_,_,_,_,_,_,a1,b1,c1,d1,e1,f1,g1,h1,_,_,_,_,_,_,_,_,i1,j1,k1,l1,m1,n1,o1,p1},{_,_,_,_,_,_,_,_,a2,b2,c2,d2,e2,f2,g2,h2,_,_,_,_,_,_,_,_,i2,j2,k2,l2,m2,n2,o2,p2})
//   -> {a2,a1,b2,b1,c2,c1,d2,d1,e2,e1,f2,f1,g2,g1,h2,h1,i2,i1,j2,j1,k2,k1,l2,l1,m2,m1,n2,n1,o2,o1,p2,p1}
[[nodiscard]] inline intNxM<int8, 32> unpacklo(intNxM<int8, 32> x, intNxM<int8, 32> y) noexcept { return _mm256_unpacklo_epi8(x, y); }

[[nodiscard]] inline int64 reduce_and(intNxM<int64, 2> v) noexcept { return (v & unpackhi(v, v)).get<0>(); }
[[nodiscard]] inline int64 reduce_or (intNxM<int64, 2> v) noexcept { return (v | unpackhi(v, v)).get<0>(); }
[[nodiscard]] inline int64 reduce_xor(intNxM<int64, 2> v) noexcept { return (v ^ unpackhi(v, v)).get<0>(); }
[[nodiscard]] inline int64 reduce_add(intNxM<int64, 2> v) noexcept { return (v + unpackhi(v, v)).get<0>(); }
[[nodiscard]] inline int32 reduce_and(intNxM<int32, 4> v) noexcept { auto t = v.get_int64<0>() & v.get_int64<1>(); return static_cast<int32>(t & (t >> 32)); }
[[nodiscard]] inline int32 reduce_or (intNxM<int32, 4> v) noexcept { auto t = v.get_int64<0>() | v.get_int64<1>(); return static_cast<int32>(t | (t >> 32)); }
[[nodiscard]] inline int32 reduce_xor(intNxM<int32, 4> v) noexcept { auto t = v.get_int64<0>() ^ v.get_int64<1>(); return static_cast<int32>(t ^ (t >> 32)); }
[[nodiscard]] inline int32 reduce_add(intNxM<int32, 4> v) noexcept { auto t = v.get_int64<0>() + v.get_int64<1>(); return static_cast<int32>(t + (t >> 32)); }
[[nodiscard]] inline int16 reduce_and(intNxM<int16, 8> v) noexcept { auto t = v.get_int64<0>() & v.get_int64<1>(); t &= (t >> 32); return static_cast<int16>(t & (t >> 16)); }
[[nodiscard]] inline int16 reduce_or (intNxM<int16, 8> v) noexcept { auto t = v.get_int64<0>() | v.get_int64<1>(); t |= (t >> 32); return static_cast<int16>(t | (t >> 16)); }
[[nodiscard]] inline int16 reduce_xor(intNxM<int16, 8> v) noexcept { auto t = v.get_int64<0>() ^ v.get_int64<1>(); t ^= (t >> 32); return static_cast<int16>(t ^ (t >> 16)); }
[[nodiscard]] inline int16 reduce_add(intNxM<int16, 8> v) noexcept { auto t = v.get_int64<0>() + v.get_int64<1>(); t += (t >> 32); return static_cast<int16>(t + (t >> 16)); }
[[nodiscard]] inline int8  reduce_and(intNxM<int8 ,16> v) noexcept { auto t = v.get_int64<0>() & v.get_int64<1>(); t &= (t >> 32); t &= (t >> 16); return static_cast<int8>(t & (t >> 8)); }
[[nodiscard]] inline int8  reduce_or (intNxM<int8 ,16> v) noexcept { auto t = v.get_int64<0>() | v.get_int64<1>(); t |= (t >> 32); t |= (t >> 16); return static_cast<int8>(t | (t >> 8)); }
[[nodiscard]] inline int8  reduce_xor(intNxM<int8 ,16> v) noexcept { auto t = v.get_int64<0>() ^ v.get_int64<1>(); t ^= (t >> 32); t ^= (t >> 16); return static_cast<int8>(t ^ (t >> 8)); }
[[nodiscard]] inline int8  reduce_add(intNxM<int8 ,16> v) noexcept { auto t = v.get_int64<0>() + v.get_int64<1>(); t += (t >> 32); t += (t >> 16); return static_cast<int8>(t + (t >> 8)); }

[[nodiscard]] inline int64 reduce_and(intNxM<int64, 4> v) noexcept { return reduce_and(v.get_int128<0>() & v.get_int128<1>()); }
[[nodiscard]] inline int64 reduce_or (intNxM<int64, 4> v) noexcept { return reduce_or (v.get_int128<0>() | v.get_int128<1>()); }
[[nodiscard]] inline int64 reduce_xor(intNxM<int64, 4> v) noexcept { return reduce_xor(v.get_int128<0>() ^ v.get_int128<1>()); }
[[nodiscard]] inline int64 reduce_add(intNxM<int64, 4> v) noexcept { return reduce_add(v.get_int128<0>() + v.get_int128<1>()); }
[[nodiscard]] inline int32 reduce_and(intNxM<int32, 8> v) noexcept { return reduce_and(v.get_int128<0>() & v.get_int128<1>()); }
[[nodiscard]] inline int32 reduce_or (intNxM<int32, 8> v) noexcept { return reduce_or (v.get_int128<0>() | v.get_int128<1>()); }
[[nodiscard]] inline int32 reduce_xor(intNxM<int32, 8> v) noexcept { return reduce_xor(v.get_int128<0>() ^ v.get_int128<1>()); }
[[nodiscard]] inline int32 reduce_add(intNxM<int32, 8> v) noexcept { return reduce_add(v.get_int128<0>() + v.get_int128<1>()); }
[[nodiscard]] inline int16 reduce_and(intNxM<int16,16> v) noexcept { return reduce_and(v.get_int128<0>() & v.get_int128<1>()); }
[[nodiscard]] inline int16 reduce_or (intNxM<int16,16> v) noexcept { return reduce_or (v.get_int128<0>() | v.get_int128<1>()); }
[[nodiscard]] inline int16 reduce_xor(intNxM<int16,16> v) noexcept { return reduce_xor(v.get_int128<0>() ^ v.get_int128<1>()); }
[[nodiscard]] inline int16 reduce_add(intNxM<int16,16> v) noexcept { return reduce_add(v.get_int128<0>() + v.get_int128<1>()); }
[[nodiscard]] inline int8  reduce_and(intNxM<int8 ,32> v) noexcept { return reduce_and(v.get_int128<0>() & v.get_int128<1>()); }
[[nodiscard]] inline int8  reduce_or (intNxM<int8 ,32> v) noexcept { return reduce_or (v.get_int128<0>() | v.get_int128<1>()); }
[[nodiscard]] inline int8  reduce_xor(intNxM<int8 ,32> v) noexcept { return reduce_xor(v.get_int128<0>() ^ v.get_int128<1>()); }
[[nodiscard]] inline int8  reduce_add(intNxM<int8 ,32> v) noexcept { return reduce_add(v.get_int128<0>() + v.get_int128<1>()); }

#endif
