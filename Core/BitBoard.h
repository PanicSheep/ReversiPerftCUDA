#pragma once
#include <intrin.h>
#include <cassert>
#include <cstdint>
#include "Bit.h"

enum class Field : uint8_t
{
	A1, B1, C1, D1, E1, F1, G1, H1,
	A2, B2, C2, D2, E2, F2, G2, H2,
	A3, B3, C3, D3, E3, F3, G3, H3,
	A4, B4, C4, D4, E4, F4, G4, H4,
	A5, B5, C5, D5, E5, F5, G5, H5,
	A6, B6, C6, D6, E6, F6, G6, H6,
	A7, B7, C7, D7, E7, F7, G7, H7,
	A8, B8, C8, D8, E8, F8, G8, H8,
	invalid
};

// An 8x8 board of binary integers.
class BitBoard
{
	uint64_t b{};
public:
	constexpr BitBoard() noexcept = default;
	CUDA_CALLABLE constexpr BitBoard(uint64_t b) noexcept : b(b) {}
	CUDA_CALLABLE constexpr explicit BitBoard(Field f) noexcept : BitBoard(1ULL << static_cast<uint8>(f)) {}

	static constexpr BitBoard Bit(int x, int y) noexcept { return 1ULL << (x + y * 8); }
	static constexpr BitBoard HorizontalLine(int i) noexcept { return 0xFFULL << (i * 8); }
	static constexpr BitBoard VerticalLine(int i) noexcept { return 0x0101010101010101ULL << i; }
	static constexpr BitBoard UpperDiagonalLine(int i) noexcept { return 0x8040201008040201ULL << (i * 8); }
	static constexpr BitBoard LowerDiagonalLine(int i) noexcept { return 0x8040201008040201ULL >> (i * 8); }
	static constexpr BitBoard UpperCodiagonalLine(int i) noexcept { return 0x0102040810204080ULL << (i * 8); }
	static constexpr BitBoard LowerCodiagonalLine(int i) noexcept { return 0x0102040810204080ULL >> (i * 8); }
	static constexpr BitBoard DiagonalLine(int i) noexcept { return (i > 0) ? UpperDiagonalLine(i) : LowerDiagonalLine(-i); }
	static constexpr BitBoard CodiagonalLine(int i) noexcept { return (i > 0) ? UpperCodiagonalLine(i) : LowerCodiagonalLine(-i); }
	static constexpr BitBoard Corners() noexcept { return 0x8100000000000081ULL; }
	static constexpr BitBoard Edges() noexcept { return 0xFF818181818181FFULL; }
	static constexpr BitBoard LeftHalf() noexcept { return 0xF0F0F0F0F0F0F0F0ULL; }
	static constexpr BitBoard RightHalf() noexcept { return 0x0F0F0F0F0F0F0F0FULL; }
	static constexpr BitBoard UpperHalf() noexcept { return 0xFFFFFFFF00000000ULL; }
	static constexpr BitBoard LowerHalf() noexcept { return 0x00000000FFFFFFFFULL; }
	static constexpr BitBoard StrictlyLeftUpper() noexcept { return 0xFEFCF8F0E0C08000ULL; }
	static constexpr BitBoard StrictlyLeftLower() noexcept { return 0x0080C0E0F0F8FCFEULL; }
	static constexpr BitBoard StrictlyRighUppert() noexcept { return 0x7F3F1F0F07030100ULL; }
	static constexpr BitBoard StrictlyRighLowert() noexcept { return 0x000103070F1F3F7FULL; }

	CUDA_CALLABLE constexpr operator uint64_t() const noexcept { return b; }

	CUDA_CALLABLE constexpr BitBoard operator~() const noexcept { return ~b; }
	BitBoard& operator&=(const BitBoard& o) noexcept { b &= o.b; return *this; }
	BitBoard& operator|=(const BitBoard& o) noexcept { b |= o.b; return *this; }
	BitBoard& operator^=(const BitBoard& o) noexcept { b ^= o.b; return *this; }
	BitBoard& operator&=(uint64_t o) noexcept { b &= o; return *this; }
	BitBoard& operator|=(uint64_t o) noexcept { b |= o; return *this; }
	BitBoard& operator^=(uint64_t o) noexcept { b ^= o; return *this; }
	friend CUDA_CALLABLE constexpr BitBoard operator&(const BitBoard& l, const BitBoard& r) noexcept { return l.b & r.b; }
	friend CUDA_CALLABLE constexpr BitBoard operator|(const BitBoard& l, const BitBoard& r) noexcept { return l.b | r.b; }
	friend CUDA_CALLABLE constexpr BitBoard operator^(const BitBoard& l, const BitBoard& r) noexcept { return l.b ^ r.b; }
	friend CUDA_CALLABLE constexpr BitBoard operator&(uint64_t l, const BitBoard& r) noexcept { return l & r.b; }
	friend CUDA_CALLABLE constexpr BitBoard operator|(uint64_t l, const BitBoard& r) noexcept { return l | r.b; }
	friend CUDA_CALLABLE constexpr BitBoard operator^(uint64_t l, const BitBoard& r) noexcept { return l ^ r.b; }
	friend CUDA_CALLABLE constexpr BitBoard operator&(const BitBoard& l, uint64_t r) noexcept { return l.b & r; }
	friend CUDA_CALLABLE constexpr BitBoard operator|(const BitBoard& l, uint64_t r) noexcept { return l.b | r; }
	friend CUDA_CALLABLE constexpr BitBoard operator^(const BitBoard& l, uint64_t r) noexcept { return l.b ^ r; }

	//[[nodiscard]] constexpr auto operator<=>(const BitBoard&) const noexcept = default;
	[[nodiscard]] CUDA_CALLABLE constexpr bool operator==(const BitBoard& o) const noexcept { return b == o.b; }
	[[nodiscard]] CUDA_CALLABLE constexpr bool operator!=(const BitBoard& o) const noexcept { return b != o.b; }

	[[nodiscard]] bool Get(Field f) const noexcept { return b & (1ULL << static_cast<uint8>(f)); }
	[[nodiscard]] bool Get(int x, int y) const noexcept { return b & (1ULL << (x + 8 * y)); }

	constexpr void Set(Field f) noexcept { b |= 1ULL << static_cast<uint8>(f); }
	constexpr void Set(int x, int y) noexcept { b |= 1ULL << (x + 8 * y); }

	void Clear(Field f) noexcept { b &= ~(1ULL << static_cast<uint8>(f)); }
	void Clear(int x, int y) noexcept { b &= ~(1ULL << (x + 8 * y)); }

	[[nodiscard]] CUDA_CALLABLE constexpr bool empty() const noexcept { return b == 0; }

	[[nodiscard]] CUDA_CALLABLE Field FirstSet() const noexcept { return static_cast<Field>(std::countr_zero(b)); }
	CUDA_CALLABLE void ClearFirstSet() noexcept { RemoveLSB(b); }

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
};

[[nodiscard]] BitBoard FlipCodiagonal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipDiagonal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipHorizontal(BitBoard) noexcept;
[[nodiscard]] BitBoard FlipVertical(BitBoard) noexcept;

CUDA_CALLABLE inline int countl_zero(const BitBoard& b) noexcept { return std::countl_zero(static_cast<uint64_t>(b)); }
CUDA_CALLABLE inline int countl_one(const BitBoard& b) noexcept { return std::countl_one(static_cast<uint64_t>(b)); }
CUDA_CALLABLE inline int countr_zero(const BitBoard& b) noexcept { return std::countr_zero(static_cast<uint64_t>(b)); }
CUDA_CALLABLE inline int countr_one(const BitBoard& b) noexcept { return std::countr_one(static_cast<uint64_t>(b)); }
CUDA_CALLABLE inline int popcount(const BitBoard& b) noexcept { return std::popcount(static_cast<uint64_t>(b)); }

constexpr BitBoard operator""_BitBoard(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard bb;
	for (int y = 0; y < 8; y++)
		for (int x = 0; x < 8; x++)
		{
			char symbol = c[119 - 15 * y - 2 * x];
			if (symbol != '-')
				bb.Set(x, y);
		}
	return bb;
}
