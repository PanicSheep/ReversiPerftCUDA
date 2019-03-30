#pragma once
#include "MacrosHell.h"
#include <cstdint>

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
	CUDA_CALLABLE CMoves(uint64_t moves) noexcept : m_moves(moves) {}
	CUDA_CALLABLE uint64_t Get() const { return m_moves; }
	bool operator==(const CMoves& o) const noexcept { return m_moves == o.m_moves; }
	
	CUDA_CALLABLE std::size_t size() const noexcept { return PopCount(m_moves); }
	CUDA_CALLABLE bool empty() const noexcept { return m_moves == 0; }
	CUDA_CALLABLE void clear() noexcept { m_moves = 0; }
	
	CUDA_CALLABLE bool HasMove(CMove) const noexcept;
	CUDA_CALLABLE CMove PeekMove() const noexcept;
	CUDA_CALLABLE CMove ExtractMove() noexcept
	{
		const auto LSB = static_cast<CMove>(BitScanLSB(m_moves));
		RemoveLSB(m_moves);
		return LSB;
	}
	
	CUDA_CALLABLE void Remove(CMove) noexcept;
	CUDA_CALLABLE void Remove(uint64_t moves) noexcept;
	CUDA_CALLABLE void Filter(uint64_t moves) noexcept;
};
