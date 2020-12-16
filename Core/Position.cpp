#include "Position.h"

void Position::FlipCodiagonal() noexcept { P.FlipCodiagonal(); O.FlipCodiagonal(); }
void Position::FlipDiagonal  () noexcept { P.FlipDiagonal  (); O.FlipDiagonal  (); }
void Position::FlipHorizontal() noexcept { P.FlipHorizontal(); O.FlipHorizontal(); }
void Position::FlipVertical  () noexcept { P.FlipVertical  (); O.FlipVertical  (); }

void Position::FlipToUnique() noexcept
{
	Position candidate = *this;
	Position min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
	candidate.FlipHorizontal();		if (candidate < min) min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
	candidate.FlipCodiagonal();		if (candidate < min) min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
	candidate.FlipHorizontal();		if (candidate < min) min = candidate;
	candidate.FlipVertical();		if (candidate < min) min = candidate;
	*this = min;
}

Position Position::Start()
{
	return
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - O X - - -"
		"- - - X O - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;
}

Position Position::StartETH()
{
	return
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X - - -"
		"- - - O O - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"_pos;
}

BitBoard Position::ParityQuadrants() const
{
	// 4 x SHIFT, 4 x XOR, 1 x AND, 1 x NOT, 1x OR, 1 x MUL
	// = 12 OPs
	BitBoard E = Empties();
	E ^= E >> 1;
	E ^= E >> 2;
	E ^= E >> 8;
	E ^= E >> 16;
	E &= 0x0000'0011'0000'0011ULL;
	return E * 0x0000'0000'0F0F'0F0FULL;
}

Position FlipCodiagonal(Position pos) noexcept
{
	pos.FlipCodiagonal();
	return pos;
}

Position FlipDiagonal(Position pos) noexcept
{
	pos.FlipDiagonal();
	return pos;
}

Position FlipHorizontal(Position pos) noexcept
{
	pos.FlipHorizontal();
	return pos;
}

Position FlipVertical(Position pos) noexcept
{
	pos.FlipVertical();
	return pos;
}

Position FlipToUnique(Position pos) noexcept
{
	pos.FlipToUnique();
	return pos;
}

Score EvalGameOver(const Position& pos) noexcept
{
	const auto Ps = popcount(pos.Player());
	const auto Os = popcount(pos.Opponent());
	if (Ps > Os)
		return 64 - 2 * Os;
	if (Ps < Os)
		return 2 * Ps - 64;
	return 0;
}

CUDA_CALLABLE Position Play(const Position& pos, Field move, BitBoard flips)
{
	assert((pos.Opponent() & flips) == flips); // only flipping opponent stones.

	return { pos.Opponent() ^ flips, pos.Player() ^ flips ^ BitBoard(move) };
}

CUDA_CALLABLE Position Play(const Position& pos, Field move)
{
	//assert(pos.Empties().Get(move)); // move field is free.

	const auto flips = Flips(pos, move);
	return Play(pos, move, flips);
}

CUDA_CALLABLE Position PlayPass(const Position& pos) noexcept
{
	return { pos.Opponent(), pos.Player() };
}