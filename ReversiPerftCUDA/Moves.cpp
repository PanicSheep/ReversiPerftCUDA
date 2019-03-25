#include "Moves.h"

bool CMoves::HasMove(const CMove move) const noexcept
{
	return TestBit(m_moves, move);
}

CMove CMoves::PeekMove() const noexcept
{
	return static_cast<CMove>(BitScanLSB(m_moves));
}

CMove CMoves::ExtractMove() noexcept
{
	const auto LSB = static_cast<CMove>(BitScanLSB(m_moves));
	RemoveLSB(m_moves);
	return LSB;
}

void CMoves::Remove(const CMove move) noexcept
{
	if (move != CMove::invalid)
		ResetBit(m_moves, move);
}

void CMoves::Remove(uint64_t moves) noexcept
{
	m_moves &= ~moves;
}

void CMoves::Filter(uint64_t moves) noexcept
{
	m_moves &= moves;
}

CBestMoves Merge(
	CBestMoves best_moves1, int8_t depth1, uint8_t selectivity1,
	CBestMoves best_moves2, int8_t depth2, uint8_t selectivity2)
{
	if ((depth2 >= depth1) && (selectivity2 <= selectivity1))
	{
		std::swap(best_moves1, best_moves2);
		std::swap(depth1, depth2);
		std::swap(selectivity1, selectivity2);
	}

	if (best_moves1.PV == CMove::invalid)
		return best_moves2;
	if (best_moves2.PV != CMove::invalid)
	{
		if (best_moves1.AV == CMove::invalid)
			return CBestMoves(best_moves1.PV, best_moves2.PV);
		if (best_moves1.PV != best_moves2.PV)
			return CBestMoves(best_moves1.PV, best_moves2.PV);
	}
	return best_moves1;
}