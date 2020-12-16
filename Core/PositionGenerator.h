#pragma once
#include "Algorithm.h"
#include "Position.h"
#include "Moves.h"
#include "Player.h"

#include <random>
#include <stack>
#include <iterator>
#include <optional>
#include <stack>

Position RandomPosition(std::mt19937_64& rnd_engine, BitBoard exclude = {});
Position RandomPosition(std::mt19937_64& rnd_engine, int empty_count);

// PositionGenerator
namespace PosGen
{
	// Generator of random Position.
	class Random
	{
		BitBoard mask;
		std::mt19937_64 rnd_engine;
	public:
		Random(uint64_t seed = std::random_device{}(), BitBoard exclude = {}) : mask(~exclude), rnd_engine(seed) {}

		Position operator()();
	};

	// Generator of random Position with given empty count.
	class Random_with_empty_count
	{
		const std::size_t empty_count;
		std::mt19937_64 rnd_engine;
	public:
		Random_with_empty_count(std::size_t empty_count, uint64_t seed = std::random_device{}()) : empty_count(empty_count), rnd_engine(seed) {}
		
		Position operator()();
	};
	
	// Generator of played Position with given empty count.
	class Played
	{
		Player &first, &second;
		const std::size_t empty_count;
		const Position start;
	public:
		Played(Player& first, Player& second, std::size_t empty_count, Position start = Position::Start());

		Position operator()();
	};
}

class ChildrenGenerator
{
	class Iterator
	{
		struct PosMov {
			Position pos;
			Moves moves;
		};
		const int plies = 0;
		const bool pass_is_a_ply = 0;
		std::stack<PosMov> stack{};
		Position current;

		bool try_push(const Position& pos);
		bool stack_is_full() const { return stack.size() == plies; }
		void pop_exhausted_elements();
		void try_to_populate(); // fills stack and generates 'current' or leaves stack empty.
	public:
		using difference_type = std::ptrdiff_t;
		using value_type = Position;
		using reference = Position&;
		using pointer = Position*;
		using iterator_category = std::input_iterator_tag;

		Iterator() noexcept = default;
		Iterator(Position start, int plies, bool pass_is_a_ply);

		[[nodiscard]] bool operator==(const Iterator&) const noexcept { return false; }
		[[nodiscard]] bool operator!=(const Iterator&) const noexcept { return !stack.empty(); }
		Iterator& operator++();
		[[nodiscard]] Position& operator*();
		[[nodiscard]] const Position& operator*() const;
	};

	Position start;
	const int plies;
	const bool pass_is_a_ply;
public:
	ChildrenGenerator(Position start, int plies, bool pass_is_a_ply)
		: start(start), plies(plies), pass_is_a_ply(pass_is_a_ply)
	{}

	[[nodiscard]] Iterator begin() const { return {start, plies, pass_is_a_ply}; }
	[[nodiscard]] Iterator cbegin() const { return {start, plies, pass_is_a_ply}; }
	[[nodiscard]] static Iterator end() { return {}; }
	[[nodiscard]] static Iterator cend() { return {}; }
};

ChildrenGenerator Children(Position, int plies, bool pass_is_a_ply);
ChildrenGenerator Children(Position, int empty_count);