#include "Moves.h"

bool CMoves::HasMove(const CMove move) const noexcept
{
	return TestBit(m_moves, move);
}

CMove CMoves::PeekMove() const noexcept
{
	return static_cast<CMove>(BitScanLSB(m_moves));
}

//CMove CMoves::ExtractMove() noexcept
//{
//	const auto LSB = static_cast<CMove>(BitScanLSB(m_moves));
//	RemoveLSB(m_moves);
//	return LSB;
//}

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
