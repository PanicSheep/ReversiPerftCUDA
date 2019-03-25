#include "Perft_basic.h"
#include "Utility.h"

NumberOfGamesCalculator::NumberOfGamesCalculator(uint64_t BytesRAM, HyperParameter hyperparameter)
	: ht(BytesRAM / sizeof(HashTablePerft::nodetype))
	, hyperparameter(hyperparameter)
{}

uint64_t NumberOfGamesCalculator::CorrectValue(const uint8_t depth)
{
	const uint64_t correct[] = { 1, 4, 12, 56, 244, 1396, 8200, 55092, 390216, 3005288, 24571056, 212258216, 1939879668, 18429618408, 184041761768, 1891831332208, 20301171282452, 222742563853912, 2534535926617852, 29335558770589276, 20, 21 };
	return correct[depth];
}

uint64_t NumberOfGamesCalculator::Calc(const uint8_t depth)
{
	if (depth == 0)
		return perft_0();

	CPosition start = CPosition::StartPosition();
	assert(start == FlipDiagonal(start));
	assert(start == FlipCodiagonal(start));
	const uint64_t symmetries = 4;
	// Because 'start' has a fourfold symmetrie only one symmetry needs to be calculated.

	CPosition pos = start.Play(start.PossibleMoves().ExtractMove());
	return symmetries * Calc(pos, depth - 1);
}

uint64_t NumberOfGamesCalculator::Calc(const CPosition& pos, const uint8_t depth)
{
	if (depth < hyperparameter.uniqueness_optimization_depth)
		return perft(pos, depth);

	fill_unique_pos_multiplier(pos, hyperparameter.initial_depth);
	DegeneracyAccessor accessor(unique_pos_multiplier);

	const auto size = static_cast<int64_t>(accessor.size());
	std::size_t sum = 0;
	#pragma omp parallel for schedule(dynamic,1) reduction(+:sum)
	for (int64_t i = 0; i < size; i++)
		sum += perft(accessor.Position(i), depth - hyperparameter.initial_depth) * accessor.Degeneracy(i);
	return sum;
}

uint64_t NumberOfGamesCalculator::perft(const CPosition& pos, const uint8_t depth)
{
	switch (depth)
	{
		case 0: return perft_0();
		case 1: return perft_1(pos);
		case 2: return perft_2(pos);
		default:
			return perft_HT(pos, depth);
	}
}

uint64_t NumberOfGamesCalculator::perft_HT(const CPosition& pos, uint8_t depth)
{
	if (depth == 2)
		return perft_2(pos);

	auto moves = pos.PossibleMoves();

	if (moves.empty())
	{
		const auto PosPass = pos.PlayPass();
		if (PosPass.HasMoves())
			return perft_HT(PosPass, depth - 1);
		return 0;
	}

	if (const auto ret = ht.LookUp(PerftKey(pos, depth)); ret.has_value())
		return ret.value();

	uint64_t sum = 0;
	while (!moves.empty())
		sum += perft_HT(pos.Play(moves.ExtractMove()), depth - 1);

	ht.Update(PerftKey(pos, depth), sum);
	return sum;
}

// perft for 2 plies left
uint64_t NumberOfGamesCalculator::perft_2(const CPosition& pos)
{
	auto moves = pos.PossibleMoves();

	if (moves.empty())
		return perft_1(pos.PlayPass());

	uint64_t sum = 0ui64;
	while (!moves.empty())
	{
		const auto next_pos = pos.Play(moves.ExtractMove());
		const auto moves2 = next_pos.PossibleMoves();
		if (!moves2.empty())
			sum += moves2.size();
		else
			sum += static_cast<uint64_t>(next_pos.PlayPass().HasMoves());
	}

	return sum;
}

// perft for 1 ply left
uint64_t NumberOfGamesCalculator::perft_1(const CPosition& pos)
{
	return pos.PossibleMoves().size();
}

// perft for 0 plies left
uint64_t NumberOfGamesCalculator::perft_0()
{
	return 1;
}

void NumberOfGamesCalculator::fill_unique_pos_multiplier(CPosition pos, const uint8_t depth)
{
	if (depth == 0) {
		pos.FlipToMin();
		unique_pos_multiplier.Insert(pos);
		return;
	}

	auto moves = pos.PossibleMoves();

	if (moves.empty())
	{
		auto PosPass = pos.PlayPass();
		if (PosPass.HasMoves())
			fill_unique_pos_multiplier(PosPass, depth - 1);
		return;
	}

	while (!moves.empty())
		fill_unique_pos_multiplier(pos.Play(moves.ExtractMove()), depth - 1);
}

NumberOfGamesCalculator::HyperParameter::HyperParameter(uint8_t uniqueness_optimization_depth, std::size_t initial_depth, uint8_t gpu)
	: uniqueness_optimization_depth(uniqueness_optimization_depth)
	, initial_depth(initial_depth)
	, gpu(gpu)
{}
