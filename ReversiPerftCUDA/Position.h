#pragma once
#include "MacrosHell.h"
#include "Moves.h"

class CPosition
{
	uint64_t P{ 0 }, O{ 0 };
public:

	CPosition() = default;
	CPosition(uint64_t P, uint64_t O) : P(P), O(O) {}

	static CPosition StartPosition   () { return CPosition(0x0000000810000000ui64, 0x0000001008000000ui64); }
	static CPosition StartPositionETH() { return CPosition(0x0000001800000000ui64, 0x0000000018000000ui64); }

	uint64_t GetP() const { return P; }
	uint64_t GetO() const { return O; }

	uint64_t Parity() const; // TODO: Remove maybe.
	uint64_t GetParityQuadrants() const;
	uint64_t Empties() const       { return ~(P | O); }
	uint64_t EmptyCount() const    { return PopCount(Empties()); }
	CMoves   PossibleMoves() const;
	bool     HasMoves() const;

	CPosition Play(CMove move) const;
	CPosition Play(CMove move, uint64_t flips) const;
	CPosition PlayPass() const { return CPosition(O, P); }

	void FlipCodiagonal();
	void FlipDiagonal  ();
	void FlipHorizontal();
	void FlipVertical  ();
	void FlipToMin     ();

	bool operator==(const CPosition& o) const { return (P == o.P) && (O == o.O); }
	bool operator!=(const CPosition& o) const { return (P != o.P) || (O != o.O); }
	bool operator<=(const CPosition& o) const { return (P <= o.P) || ((P == o.P) && (O <= o.O)); }
	bool operator>=(const CPosition& o) const { return (P >= o.P) || ((P == o.P) && (O >= o.O)); }
	bool operator< (const CPosition& o) const { return (P <  o.P) || ((P == o.P) && (O <  o.O)); }
	bool operator> (const CPosition& o) const { return (P >  o.P) || ((P == o.P) && (O >  o.O)); }
};

inline CPosition FlipCodiagonal(CPosition pos) { pos.FlipCodiagonal(); return pos; }
inline CPosition FlipDiagonal  (CPosition pos) { pos.FlipDiagonal  (); return pos; }
inline CPosition FlipHorizontal(CPosition pos) { pos.FlipHorizontal(); return pos; }
inline CPosition FlipVertical  (CPosition pos) { pos.FlipVertical  (); return pos; }

namespace std
{
	template<> struct hash<CPosition>
	{
		std::size_t operator()(const CPosition& pos) const
		{
			uint64_t P = pos.GetP();
			uint64_t O = pos.GetO();
			P ^= P >> 36;
			O ^= O >> 21;
			return P * O;
		}
	};
}