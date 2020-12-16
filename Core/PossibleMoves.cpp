#include "Bit.h"
#include "Position.h"

CUDA_CALLABLE Moves PossibleMoves(const Position& pos) noexcept
{
	#ifdef __NVCC__
		return detail::PossibleMoves_x64(pos);
	#elif defined(__AVX512F__)
		return detail::PossibleMoves_AVX512(pos);
	#elif defined(__AVX2__)
		return detail::PossibleMoves_AVX2(pos);
	#else
		return detail::PossibleMoves_x64(pos);
	#endif
}

#ifdef __AVX512F__
Moves detail::PossibleMoves_AVX512(const Position& pos) noexcept
{
	// 6 x SHIFT, 7 x AND, 5 x OR, 1 x NOT
	// = 19 OPs

	// 1 x AND
	const __m512i maskO = _mm512_set1_epi64(pos.Opponent()) & _mm512_set_epi64(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL, 0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
	const __m512i shift1 = _mm512_set_epi64(1, 8, 7, 9, -1, -8, -7, -9);
	const __m512i shift2 = shift1 + shift1;

	// 6 x SHIFT, 5 x AND, 3 x OR
	__m512i flip = maskO & _mm512_rolv_epi64(_mm512_set1_epi64(pos.Player()), shift1);
	__m512i mask = maskO & _mm512_rolv_epi64(maskO, shift1);
	flip |= maskO & _mm512_rolv_epi64(flip, shift1);
	flip |= mask & _mm512_rolv_epi64(flip, shift2);
	flip |= mask & _mm512_rolv_epi64(flip, shift2);
	flip = _mm512_rolv_epi64(flip, shift1);

	// 1 x NOT, 2 x OR, 1 x AND
	return ~(P | O) & _mm512_reduce_or_epi64(flip);
}
#endif

#ifdef __AVX2__
Moves detail::PossibleMoves_AVX2(const Position& pos) noexcept
{
	// 8 x OR, 12 x SHIFT, 11 x AND, 1 x NOT, 1 x reduce_or
	// = 22 OPs + reduce_or

	// 1 x AND
	const int64x4 maskO = int64x4(pos.Opponent()) & int64x4(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
	const int64x4 P(pos.Player());

	// 2 x SHIFT, 2 x AND
	int64x4 flip1 = maskO & (P << int64x4(1, 8, 7, 9));
	int64x4 flip2 = maskO & (P >> int64x4(1, 8, 7, 9));

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 |= maskO & (flip1 << int64x4(1, 8, 7, 9));
	flip2 |= maskO & (flip2 >> int64x4(1, 8, 7, 9));

	// 2 x SHIFT, 1 x AND
	const int64x4 mask1 = maskO & (maskO << int64x4(1, 8, 7, 9));
	const int64x4 mask2 = mask1 >> int64x4(1, 8, 7, 9);

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 |= mask1 & (flip1 << int64x4(2, 16, 14, 18));
	flip2 |= mask2 & (flip2 >> int64x4(2, 16, 14, 18));

	// 2 x SHIFT, 2 x AND, 2 x OR
	flip1 |= mask1 & (flip1 << int64x4(2, 16, 14, 18));
	flip2 |= mask2 & (flip2 >> int64x4(2, 16, 14, 18));

	// 2 x SHIFT
	flip1 <<= int64x4(1, 8, 7, 9);
	flip2 >>= int64x4(1, 8, 7, 9);
	
	// 1 x NOT, 2 x OR, 1 x AND, 1 x function call
	return pos.Empties() & static_cast<uint64>(reduce_or(flip1 | flip2));
}
#endif

CUDA_CALLABLE Moves detail::PossibleMoves_x64(const Position& pos) noexcept
{
	// 48 x SHIFT, 42 x AND, 32 x OR, 1 x NOT
	// = 123 OPs
	
	// 1 x AND
	const uint64_t maskO = pos.Opponent() & 0x7E7E7E7E7E7E7E7EULL;

	// 8 x SHIFT, 8 x AND
	uint64_t flip1 = maskO & (pos.Player() << 1);
	uint64_t flip2 = maskO & (pos.Player() >> 1);
	uint64_t flip3 = pos.Opponent() & (pos.Player() << 8);
	uint64_t flip4 = pos.Opponent() & (pos.Player() >> 8);
	uint64_t flip5 = maskO & (pos.Player() << 7);
	uint64_t flip6 = maskO & (pos.Player() >> 7);
	uint64_t flip7 = maskO & (pos.Player() << 9);
	uint64_t flip8 = maskO & (pos.Player() >> 9);

	// 8 x SHIFT, 8 x AND, 8 x OR
	flip1 |= maskO & (flip1 << 1);
	flip2 |= maskO & (flip2 >> 1);
	flip3 |= pos.Opponent() & (flip3 << 8);
	flip4 |= pos.Opponent() & (flip4 >> 8);
	flip5 |= maskO & (flip5 << 7);
	flip6 |= maskO & (flip6 >> 7);
	flip7 |= maskO & (flip7 << 9);
	flip8 |= maskO & (flip8 >> 9);

	// 8 x SHIFT, 8 x AND
	uint64_t mask1 = maskO & (maskO << 1);
	uint64_t mask2 = mask1 >> 1;
	uint64_t mask3 = pos.Opponent() & (pos.Opponent() << 8);
	uint64_t mask4 = mask3 >> 8;
	uint64_t mask5 = maskO & (maskO << 7);
	uint64_t mask6 = mask5 >> 7;
	uint64_t mask7 = maskO & (maskO << 9);
	uint64_t mask8 = mask7 >> 9;

	// 8 x SHIFT, 8 x AND, 8 x OR
	flip1 |= mask1 & (flip1 << 2);
	flip2 |= mask2 & (flip2 >> 2);
	flip3 |= mask3 & (flip3 << 16);
	flip4 |= mask4 & (flip4 >> 16);
	flip5 |= mask5 & (flip5 << 14);
	flip6 |= mask6 & (flip6 >> 14);
	flip7 |= mask7 & (flip7 << 18);
	flip8 |= mask8 & (flip8 >> 18);

	// 8 x SHIFT, 8 x AND, 8 x OR
	flip1 |= mask1 & (flip1 << 2);
	flip2 |= mask2 & (flip2 >> 2);
	flip3 |= mask3 & (flip3 << 16);
	flip4 |= mask4 & (flip4 >> 16);
	flip5 |= mask5 & (flip5 << 14);
	flip6 |= mask6 & (flip6 >> 14);
	flip7 |= mask7 & (flip7 << 18);
	flip8 |= mask8 & (flip8 >> 18);

	// 8 x SHIFT
	flip1 <<= 1;
	flip2 >>= 1;
	flip3 <<= 8;
	flip4 >>= 8;
	flip5 <<= 7;
	flip6 >>= 7;
	flip7 <<= 9;
	flip8 >>= 9;

	return pos.Empties() // 1 x AND, 1 x OR, 1 x NOT
		& (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8); // 7 x OR
}
