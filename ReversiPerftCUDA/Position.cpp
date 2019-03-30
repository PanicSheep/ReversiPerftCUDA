#include "Position.h"
#include "FlipFast.h"
#include "Utility.h"

namespace
{
	template <const int dir>
	uint64_t get_some_moves(const uint64_t P, const uint64_t mask)
	{
		// kogge-stone parallel prefix
		// 12 x SHIFT, 9 x AND, 7 x OR
		// = 28 OPs
		uint64_t flip_l, flip_r;
		uint64_t mask_l, mask_r;

		flip_l = mask & (P << dir);
		flip_r = mask & (P >> dir);

		flip_l |= mask & (flip_l << dir);
		flip_r |= mask & (flip_r >> dir);

		mask_l = mask & (mask << dir);	mask_r = mask_l >> dir;

		flip_l |= mask_l & (flip_l << (dir * 2));
		flip_r |= mask_r & (flip_r >> (dir * 2));

		flip_l |= mask_l & (flip_l << (dir * 2));
		flip_r |= mask_r & (flip_r >> (dir * 2));

		flip_l <<= dir;
		flip_r >>= dir;

		return flip_l | flip_r;
	}
}

void CPosition::FlipCodiagonal() { P = ::FlipCodiagonal(P); O = ::FlipCodiagonal(O); }
void CPosition::FlipDiagonal() { P = ::FlipDiagonal(P); O = ::FlipDiagonal(O); }
void CPosition::FlipHorizontal() { P = ::FlipHorizontal(P); O = ::FlipHorizontal(O); }
void CPosition::FlipVertical() { P = ::FlipVertical(P); O = ::FlipVertical(O); }

void CPosition::FlipToMin()
{
	CPosition cpy = *this;
	cpy.FlipVertical();		if (cpy < *this) *this = cpy;
	cpy.FlipHorizontal();	if (cpy < *this) *this = cpy;
	cpy.FlipVertical();		if (cpy < *this) *this = cpy;
	cpy.FlipDiagonal();		if (cpy < *this) *this = cpy;
	cpy.FlipVertical();		if (cpy < *this) *this = cpy;
	cpy.FlipHorizontal();	if (cpy < *this) *this = cpy;
	cpy.FlipVertical();		if (cpy < *this) *this = cpy;
}

uint64_t CPosition::Parity() const
{
	uint64_t E = Empties();
	E ^= E >> 1;
	E ^= E >> 2;
	E ^= E >> 8;
	E ^= E >> 16;
#ifdef HAS_PEXT
	return PExt(E, 0x0000001100000011ui64);
#else
	E &= 0x0000001100000011ui64;
	E |= E >> 3;
	E |= E >> 30;
	return E & 0xFui64;
#endif
}

uint64_t CPosition::GetParityQuadrants() const
{
	// 4 x SHIFT, 4 x XOR, 1 x AND, 1 x NOT, 1x OR, 1 x MUL
	// = 12 OPs
	uint64_t E = Empties();
	E ^= E >> 1;
	E ^= E >> 2;
	E ^= E >> 8;
	E ^= E >> 16;
	E &= 0x0000001100000011ui64;
	return E * 0x000000000F0F0F0Fui64;
}

CMoves CPosition::PossibleMoves() const
{
	return ::PossibleMoves(*this);
}

bool CPosition::HasMoves() const
{
	return !PossibleMoves().empty();
	//const uint64_t empties = ~(P | O);
	//if (get_some_moves<1>(P, O & 0x7E7E7E7E7E7E7E7Eui64) & empties) return true;
	//if (get_some_moves<8>(P, O & 0x00FFFFFFFFFFFF00ui64) & empties) return true;
	//if (get_some_moves<7>(P, O & 0x007E7E7E7E7E7E00ui64) & empties) return true;
	//if (get_some_moves<9>(P, O & 0x007E7E7E7E7E7E00ui64) & empties) return true;
	//return false;
}

CPosition CPosition::Play(const CMove move) const
{
	const auto flips = Flip(*this, move);
	return Play(move, flips);
}

