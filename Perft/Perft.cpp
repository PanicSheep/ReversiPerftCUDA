#include "Perft.h"
#include "Core/Core.h"
#include <algorithm>
#include <omp.h>

int64 Correct(int depth)
{
	const int64 correct[] =
	{
		                     1,
		                     4,
		                    12,
		                    56,
		                   244,
		                 1'396,
		                 8'200,
		                55'092,
		               390'216,
		             3'005'288,
		            24'571'056,
		           212'258'216,
				 1'939'879'668,
				18'429'618'408,
			   184'041'761'768,
			 1'891'831'332'208,
			20'301'171'282'452,
		   222'742'563'853'912,
		 2'534'535'926'617'852,
		29'335'558'770'589'276,
		                    20,
		                    21,
		                    22,
		                    23,
		                    24,
	};
	return correct[depth];
}


int64 BasicPerft::calculate(const Position& pos, const int depth)
{
	if (depth == 0)
		return 1;

	auto moves = PossibleMoves(pos);
	if (!moves)
	{
		Position passed = PlayPass(pos);
		if (HasMoves(passed))
			return calculate(passed, depth-1);
		return 0;
	}

	int64 sum = 0;
	for (const auto& move : moves)
		sum += calculate(Play(pos, move), depth-1);
	return sum;
}

int64 BasicPerft::calculate(const int depth)
{
	if (depth == 0)
		return calculate(Position::Start(), depth);

	// Makes use of 4-fold symmetrie.
	Position pos = Position::Start();
	pos = Play(pos, PossibleMoves(pos).First());
	return 4 * calculate(pos, depth-1);
}

// for 0 plies left
int64 UnrolledPerft::calculate_0()
{
	return 1;
}

// for 1 ply left
__forceinline int64 UnrolledPerft::calculate_1(const Position& pos)
{
	auto moves = PossibleMoves(pos);
	if (moves)
		return moves.size();
	return HasMoves(PlayPass(pos)) ? 1 : 0;
}

// for 2 plies left
int64 UnrolledPerft::calculate_2(const Position& pos)
{
	auto moves = PossibleMoves(pos);
	if (!moves)
		return PossibleMoves(PlayPass(pos)).size();

	int64 sum = 0;
	for (const auto& move : moves)
		sum += calculate_1(Play(pos, move));
	return sum;
}

int64 UnrolledPerft::calculate_n(const Position& pos, const int depth)
{
	switch (depth)
	{
		case 0: return calculate_0();
		case 1: return calculate_1(pos);
		case 2: return calculate_2(pos);
		default: break;
	}

	auto moves = PossibleMoves(pos);
	if (!moves)
	{
		Position passed = PlayPass(pos);
		if (HasMoves(passed))
			return calculate_n(passed, depth-1);
		return 0;
	}

	int64 sum = 0;
	for (const auto& move : moves)
		sum += calculate_n(Play(pos, move), depth-1);
	return sum;
}

int64 UnrolledPerft::calculate(const Position& pos, const int depth)
{
	if (initial_unroll == 0 || depth - initial_unroll <= 2)
		return calculate_n(pos, depth);

	std::vector<Position> work;
	for (const auto& pos : Children(pos, initial_unroll, true))
		work.push_back(FlipToUnique(pos));

	int64 sum = 0;
	int64 size = static_cast<int64_t>(work.size());
	#pragma omp parallel for schedule(dynamic,1) reduction(+:sum)
	for (int64 i = 0; i < size; i++)
		sum += calculate_n(work[i], depth - initial_unroll);
	return sum;
}

int64 UnrolledPerft::calculate(const int depth)
{
	if (depth == 0)
		return calculate(Position::Start(), depth);

	// Makes use of 4-fold symmetrie.
	Position pos = Position::Start();
	pos = Play(pos, PossibleMoves(pos).First());
	return 4 * calculate(pos, depth-1);
}


HashTablePerft::HashTablePerft(std::size_t bytes, int initial_unroll)
	: UnrolledPerft(initial_unroll)
	, hash_table(bytes / sizeof(BigNodeHashTable::node_type))
{}

int64 HashTablePerft::calculate_n(const Position& pos, const int depth)
{
	if (depth <= 3)
		return UnrolledPerft::calculate_n(pos, depth);

	if (auto ret = hash_table.LookUp({ pos, depth }); ret.has_value())
		return ret.value();

	auto moves = PossibleMoves(pos);
	if (!moves)
	{
		Position passed = PlayPass(pos);
		if (HasMoves(passed))
			return calculate_n(passed, depth-1);
		return 0;
	}

	int64 sum = 0;
	for (const auto& move : moves)
		sum += calculate_n(Play(pos, move), depth-1);

	hash_table.Update({ pos, depth }, sum);
	return sum;
}

int64 HashTablePerft::calculate(const Position& pos, const int depth)
{
	if (initial_unroll == 0 || depth <= initial_unroll + 2)
		return calculate_n(pos, depth);

	std::vector<Position> work;
	for (const auto& pos : Children(pos, initial_unroll, true))
		work.push_back(FlipToUnique(pos));

	int64 sum = 0;
	int64 size = static_cast<int64>(work.size());
	#pragma omp parallel for schedule(dynamic,1) reduction(+:sum)
	for (int64_t i = 0; i < size; i++)
		sum += calculate_n(work[i], depth - initial_unroll);
	return sum;
}

int64 HashTablePerft::calculate(const int depth)
{
	if (depth == 0)
		return calculate(Position::Start(), depth);

	// Makes use of 4-fold symmetrie.
	Position pos = Position::Start();
	pos = Play(pos, PossibleMoves(pos).First());
	return 4 * calculate(pos, depth-1);
}
