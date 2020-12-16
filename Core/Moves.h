#pragma once
#include "BitBoard.h"

class Moves
{
	BitBoard b{};

	class Iterator
	{
		BitBoard moves{};
	public:
		constexpr Iterator() noexcept = default;
		CUDA_CALLABLE Iterator(const BitBoard& moves) : moves(moves) {}
		CUDA_CALLABLE Iterator& operator++() { moves.ClearFirstSet(); return *this; }
		[[nodiscard]] CUDA_CALLABLE Field operator*() const { return moves.FirstSet(); }

		[[nodiscard]] CUDA_CALLABLE bool operator==(const Iterator& o) const noexcept { return moves == o.moves; }
		[[nodiscard]] CUDA_CALLABLE bool operator!=(const Iterator& o) const noexcept { return moves != o.moves; }
	};
public:
	constexpr Moves() noexcept = default;
	CUDA_CALLABLE constexpr Moves(BitBoard moves) noexcept : b(moves) {}

	//[[nodiscard]] auto operator<=>(const Moves&) const noexcept = default;
	[[nodiscard]] CUDA_CALLABLE bool operator==(const Moves& o) const noexcept { return b == o.b; }
	[[nodiscard]] CUDA_CALLABLE bool operator!=(const Moves& o) const noexcept { return b != o.b; }

	[[nodiscard]] CUDA_CALLABLE operator bool() const noexcept { return b; }

	[[nodiscard]] CUDA_CALLABLE bool empty() const noexcept { return !b; }
	[[nodiscard]] CUDA_CALLABLE int size() const noexcept { return popcount(b); }
	[[nodiscard]] bool contains(Field f) const noexcept { return b.Get(f); }

	[[nodiscard]] Field First() const noexcept { return b.FirstSet(); }
	void RemoveFirst() noexcept { b.ClearFirstSet(); }
	[[nodiscard]] Field ExtractFirst() noexcept { auto first = First(); RemoveFirst(); return first; }

	void Remove(Field f) noexcept { b.Clear(f); }
	void Remove(const BitBoard& moves) noexcept { b &= ~moves; }
	void Filter(const BitBoard& moves) noexcept { b &= moves; }

	[[nodiscard]] CUDA_CALLABLE Iterator begin() const { return Iterator(b); }
	[[nodiscard]] CUDA_CALLABLE Iterator cbegin() const { return Iterator(b); }
	[[nodiscard]] CUDA_CALLABLE static Iterator end() { return {}; }
	[[nodiscard]] CUDA_CALLABLE static Iterator cend() { return {}; }
};
