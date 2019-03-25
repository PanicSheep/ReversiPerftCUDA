#pragma once
#include "MacrosHell.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

enum Field : uint8_t
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

using CMove = Field;


class CMoves
{
	uint64_t m_moves{0};
public:
	CMoves() noexcept = default;
	CMoves(uint64_t moves) noexcept : m_moves(moves) {}

	bool operator==(const CMoves& o) const noexcept { return m_moves == o.m_moves; }

	std::size_t size() const noexcept { return PopCount(m_moves); }
	bool empty() const noexcept { return m_moves == 0; }
	void clear() noexcept { m_moves = 0; }

	bool HasMove(CMove) const noexcept;
	CMove PeekMove() const noexcept;
	CMove ExtractMove() noexcept;

	void Remove(CMove) noexcept;
	void Remove(uint64_t moves) noexcept;
	void Filter(uint64_t moves) noexcept;
};

class CBestMoves
{
public: // TODO: Make sure if there's an AV, there's a PV.
	CMove PV = CMove::invalid; // TODO: Rename to first.
	CMove AV = CMove::invalid; // TODO: Rename to second.

	CBestMoves() = default;
	CBestMoves(CMove PV, CMove AV) : PV(PV), AV(AV) {}
};

CBestMoves Merge(CBestMoves, int8_t depth1, uint8_t selectivity1, CBestMoves, int8_t depth2, uint8_t selectivity2);