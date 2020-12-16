#pragma once
#include "Bit.h"
#include "BitBoard.h"
#include "Moves.h"
#include <cstdint>
#include <tuple>

using Score = int;

constexpr Score min_score{ -64 };
constexpr Score max_score{ +64 };
constexpr Score infinity{ +66 };

// A board where every field is either taken by exactly one player or is empty.
class Position
{
	BitBoard P{}, O{};
public:
	constexpr Position() noexcept = default;
	CUDA_CALLABLE constexpr Position(BitBoard P, BitBoard O) noexcept : P(P), O(O) { assert((P & O).empty()); }

	static Position Start();
	static Position StartETH();

	static constexpr Position From(BitBoard P, BitBoard O) noexcept(false)
	{
		if ((P & O).empty())
			return { P, O };
		throw;
	}

	//[[nodiscard]] constexpr auto operator<=>(const Position&) const noexcept = default;
	[[nodiscard]] constexpr bool operator==(const Position& o) const noexcept { return P == o.P && O == o.O; }
	[[nodiscard]] constexpr bool operator!=(const Position& o) const noexcept { return P != o.P || O != o.O; }
	[[nodiscard]] constexpr bool operator<(const Position& o) const noexcept { return std::tie(P, O) < std::tie(o.P, o.O); }

	[[nodiscard]] CUDA_CALLABLE BitBoard Player() const noexcept { return P; }
	[[nodiscard]] CUDA_CALLABLE BitBoard Opponent() const noexcept { return O; }

	void FlipCodiagonal() noexcept;
	void FlipDiagonal() noexcept;
	void FlipHorizontal() noexcept;
	void FlipVertical() noexcept;
	void FlipToUnique() noexcept;

	[[nodiscard]] CUDA_CALLABLE BitBoard Discs() const { return P | O; }
	[[nodiscard]] CUDA_CALLABLE BitBoard Empties() const { return ~Discs(); }
	[[nodiscard]] int EmptyCount() const { return popcount(Empties()); }

	[[nodiscard]] BitBoard ParityQuadrants() const;
};

[[nodiscard]] Position FlipCodiagonal(Position) noexcept;
[[nodiscard]] Position FlipDiagonal(Position) noexcept;
[[nodiscard]] Position FlipHorizontal(Position) noexcept;
[[nodiscard]] Position FlipVertical(Position) noexcept;
[[nodiscard]] Position FlipToUnique(Position) noexcept;

[[nodiscard]]
constexpr Position operator""_pos(const char* c, std::size_t size)
{
	assert(size == 120);

	BitBoard P{0};
	BitBoard O{0};
	for (int j = 0; j < 8; j++)
		for (int i = 0; i < 8; i++)
		{
			char symbol = c[119 - 15 * j - 2 * i];
			if (symbol == 'X')
				P.Set(i, j);
			if (symbol == 'O')
				O.Set(i, j);
		}
	return { P, O };
}

[[nodiscard]]
Score EvalGameOver(const Position&) noexcept;

[[nodiscard]]
CUDA_CALLABLE Position Play(const Position&, Field move, BitBoard flips);

[[nodiscard]]
CUDA_CALLABLE Position Play(const Position&, Field move);

[[nodiscard]]
CUDA_CALLABLE Position PlayPass(const Position&) noexcept;

[[nodiscard]]
CUDA_CALLABLE BitBoard Flips(const Position&, Field move) noexcept;

[[nodiscard]]
int CountLastFlip(const Position&, Field move) noexcept;

[[nodiscard]]
BitBoard StableEdges(const Position&);

// Stable stones of the opponent.
[[nodiscard]]
BitBoard StableStones(const Position&);

[[nodiscard]]
CUDA_CALLABLE Moves PossibleMoves(const Position&) noexcept;

namespace detail
{
	#ifdef __AVX512F__
		[[nodiscard]]
		Moves PossibleMoves_AVX512(const Position&) noexcept;
	#endif

	#ifdef __AVX2__
		[[nodiscard]]
		Moves PossibleMoves_AVX2(const Position&) noexcept;
	#endif

	[[nodiscard]]
	CUDA_CALLABLE Moves PossibleMoves_x64(const Position&) noexcept;
}

[[nodiscard]]
CUDA_CALLABLE bool HasMoves(const Position&) noexcept;

namespace detail
{
	#ifdef __AVX512F__
		[[nodiscard]]
		bool HasMoves_AVX512(const Position&) noexcept;
	#endif

	#ifdef __AVX2__
		[[nodiscard]]
		bool HasMoves_AVX2(const Position&) noexcept;
	#endif

	[[nodiscard]]
	CUDA_CALLABLE bool HasMoves_x64(const Position&) noexcept;
}