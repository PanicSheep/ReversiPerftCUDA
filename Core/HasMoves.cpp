#include "Bit.h"
#include "Position.h"

CUDA_CALLABLE bool HasMoves(const Position& pos) noexcept
{
	#if defined(__AVX512F__)
		return detail::HasMoves_AVX512(pos);
	#elif defined(__AVX2__)
		return detail::HasMoves_AVX2(pos);
	#else
		return detail::HasMoves_x64(pos);
	#endif
}

#ifdef __AVX2__
bool detail::HasMoves_AVX2(const Position& pos) noexcept
{
	const int64x4 maskO = int64x4(pos.Opponent()) & int64x4(0x7E7E7E7E7E7E7E7EI64, 0x00FFFFFFFFFFFF00I64, 0x007E7E7E7E7E7E00I64, 0x007E7E7E7E7E7E00I64);
	const int64x4 P(pos.Player());

	int64x4 flip1 = maskO & (P << int64x4(1, 8, 7, 9));
	flip1 |= maskO & (flip1 << int64x4(1, 8, 7, 9));
	const int64x4 mask1 = maskO & (maskO << int64x4(1, 8, 7, 9));
	flip1 |= mask1 & (flip1 << int64x4(2, 16, 14, 18));
	flip1 |= mask1 & (flip1 << int64x4(2, 16, 14, 18));
	flip1 <<= int64x4(1, 8, 7, 9);
	if (pos.Empties() & static_cast<uint64>(reduce_or(flip1)))
		return true;

	int64x4 flip2 = maskO & (P >> int64x4(1, 8, 7, 9));
	flip2 |= maskO & (flip2 >> int64x4(1, 8, 7, 9));
	const int64x4 mask2 = maskO & (maskO >> int64x4(1, 8, 7, 9));
	flip2 |= mask2 & (flip2 >> int64x4(2, 16, 14, 18));
	flip2 |= mask2 & (flip2 >> int64x4(2, 16, 14, 18));
	flip2 >>= int64x4(1, 8, 7, 9);
	return pos.Empties() & static_cast<uint64>(reduce_or(flip2));
}
#endif

CUDA_CALLABLE inline uint64_t get_some_moves(const uint64_t P, const uint64_t mask, const int dir)
{
	// kogge-stone parallel prefix
	// 12 x SHIFT, 10 x AND, 7 x OR
	// = 29 OPs
	uint64_t flip_l, flip_r;
	uint64_t mask_l, mask_r;

	flip_l  = mask & (P << dir);
	flip_r  = mask & (P >> dir);

	flip_l |= mask & (flip_l << dir);
	flip_r |= mask & (flip_r >> dir);

	mask_l  = mask & (mask << dir);
	mask_r  = mask & (mask >> dir);

	flip_l |= mask_l & (flip_l << (dir * 2));
	flip_r |= mask_r & (flip_r >> (dir * 2));

	flip_l |= mask_l & (flip_l << (dir * 2));
	flip_r |= mask_r & (flip_r >> (dir * 2));

	return (flip_l << dir) | (flip_r >> dir);
}

CUDA_CALLABLE bool detail::HasMoves_x64(const Position& pos) noexcept
{
	const auto P = pos.Player();
	const auto O = pos.Opponent();
	const auto empties = pos.Empties();

	if (get_some_moves(P, O & 0x7E7E7E7E7E7E7E7EULL, 1) & empties) return true;
	if (get_some_moves(P, O & 0x00FFFFFFFFFFFF00ULL, 8) & empties) return true;
	if (get_some_moves(P, O & 0x007E7E7E7E7E7E00ULL, 7) & empties) return true;
	if (get_some_moves(P, O & 0x007E7E7E7E7E7E00ULL, 9) & empties) return true;
	return false;
}