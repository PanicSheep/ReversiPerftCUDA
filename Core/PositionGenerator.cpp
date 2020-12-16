#include "PositionGenerator.h"
#include "Bit.h"

Position RandomPosition(std::mt19937_64& rnd_engine, BitBoard exclude)
{
	// Each field has a:
	//  25% chance to belong to player,
	//  25% chance to belong to opponent,
	//  50% chance to be empty.

	auto rnd = [&]() { return std::uniform_int_distribution<uint64_t>(0, -1)(rnd_engine); };
	BitBoard a = rnd() & ~exclude;
	BitBoard b = rnd() & ~exclude;
	return { a & ~b, b & ~a };
}

Position RandomPosition(std::mt19937_64& rnd_engine, int empty_count)
{
	auto dichotron = [&]() { return std::uniform_int_distribution<int>(0, 1)(rnd_engine) == 0; };

	BitBoard P = 0;
	BitBoard O = 0;
	for (int e = 64; e > empty_count; e--)
	{
		auto rnd = std::uniform_int_distribution<std::size_t>(0, e - 1)(rnd_engine);
		auto bit = BitBoard(PDep(1ULL << rnd, Position(P, O).Empties()));

		if (dichotron())
			P |= bit;
		else
			O |= bit;
	}
	return { P, O };
}

Position PosGen::Random::operator()()
{
	// Each field has a:
	//  25% chance to belong to player,
	//  25% chance to belong to opponent,
	//  50% chance to be empty.

	auto rnd = [this]() { return std::uniform_int_distribution<uint64_t>(0, 0xFFFFFFFFFFFFFFFFULL)(rnd_engine); };
	BitBoard a = rnd() & mask;
	BitBoard b = rnd() & mask;
	return { a & ~b, b & ~a };
}

Position PosGen::Random_with_empty_count::operator()()
{
	auto dichotron = [this]() { return std::uniform_int_distribution<int>(0, 1)(rnd_engine) == 0; };

	BitBoard P = 0;
	BitBoard O = 0;
	for (std::size_t e = 64; e > empty_count; e--)
	{
		auto rnd = std::uniform_int_distribution<std::size_t>(0, e - 1)(rnd_engine);
		auto bit = BitBoard(PDep(1ULL << rnd, Position(P, O).Empties()));

		if (dichotron())
			P |= bit;
		else
			O |= bit;
	}
	return { P, O };
}

PosGen::Played::Played(Player& first, Player& second, std::size_t empty_count, Position start)
	: first(first), second(second), empty_count(empty_count), start(start)
{
	if (start.EmptyCount() < empty_count)
		throw;
}

Position PosGen::Played::operator()()
{
	Position pos = start;
	if (pos.EmptyCount() == empty_count)
		return pos;

	while (true)
	{
		Position old = pos;

		pos = first.Play(pos);
		if (pos.EmptyCount() == empty_count)
			return pos;

		pos = second.Play(pos);
		if (pos.EmptyCount() == empty_count)
			return pos;

		if (old == pos) // both players passed
			pos = start; // restart
	}
}


ChildrenGenerator::Iterator::Iterator(Position start, int plies, bool pass_is_a_ply)
	: plies(plies)
	, pass_is_a_ply(pass_is_a_ply)
{
	bool pushed = try_push(start);
	if (pushed)
		try_to_populate();
}

bool ChildrenGenerator::Iterator::try_push(const Position& pos)
{
	Moves pm = PossibleMoves(pos);
	if (pm)
	{
		stack.push({ pos, pm });
		return true;
	}
	else
	{
		Position passed = PlayPass(pos);
		Moves passed_pm = PossibleMoves(passed);
		if (passed_pm) // passing is possible, because opponent can play.
		{
			if (pass_is_a_ply)
				stack.push({ pos, Moves{0} });
			else
				stack.push({ passed, passed_pm });
			return true;
		}
		else // 'pos' is a terminal position
			return false;
	}
}

void ChildrenGenerator::Iterator::pop_exhausted_elements()
{
	while (!stack.empty() && stack.top().moves.empty())
		stack.pop();
}

void ChildrenGenerator::Iterator::try_to_populate()
{
	assert(!stack.empty());

	while (!stack_is_full())
	{
		auto& pos = stack.top().pos;
		auto& moves = stack.top().moves;
		if (moves)
		{
			Field move = moves.ExtractFirst();
			Position next = Play(pos, move);
			bool pushed = try_push(next);
			if (pushed)
				continue;
			else // 'next' is a terminal position
			{
				if (moves)
					continue;
				else
				{
					pop_exhausted_elements();
					if (stack.empty())
						return;
					else
						continue;
				}
			}
		}
		else
		{
			Position next = PlayPass(pos);
			bool pushed = try_push(next);
			assert(pushed);
		}
	}
	assert(stack_is_full());
	auto& pos = stack.top().pos;
	auto& moves = stack.top().moves;
	if (moves)
	{
		auto move = moves.ExtractFirst();
		current = Play(pos, move);
	}
	else
		current = PlayPass(pos);
}

ChildrenGenerator::Iterator& ChildrenGenerator::Iterator::operator++()
{
	assert(stack_is_full());
	pop_exhausted_elements();
	if (!stack.empty())
		try_to_populate();
	return *this;
}

Position& ChildrenGenerator::Iterator::operator*()
{
	assert(stack_is_full());
	return current;
}

const Position& ChildrenGenerator::Iterator::operator*() const
{
	assert(stack_is_full());
	return current;
}

ChildrenGenerator Children(Position start, int plies, bool pass_is_a_ply)
{
	assert(plies > 0);
	return {start, plies, pass_is_a_ply};
}

ChildrenGenerator Children(Position start, int empty_count)
{
	assert(start.EmptyCount() > empty_count);
	return {start, start.EmptyCount() - empty_count, false};
}
