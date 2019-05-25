#include "Perft.h"
#include "Utility.h"
#include "kernel.cuh"
#include <iterator>

void generate_children(CPosition pos, uint8_t depth, std::function<void(CPosition)> fkt)
{
	if (depth == 0) {
		fkt(pos);
		return;
	}

	auto moves = pos.PossibleMoves();
	if (moves.empty())
	{
		auto PosPass = pos.PlayPass();
		if (PosPass.HasMoves())
			generate_children(PosPass, depth - 1, fkt);
		return;
	}

	while (!moves.empty())
		generate_children(pos.Play(moves.ExtractMove()), depth - 1, fkt);
}

Degeneracies::Degeneracies(CPosition pos, uint8_t depth)
{
	std::map<CPosition, uint64_t> map;
	generate_children(pos, depth, [&map](CPosition pos) { map[FlipToMin(pos)]++; });
	vec = { map.begin(), map.end() };
}

NumberOfGamesCalculator::NumberOfGamesCalculator(uint64_t BytesRAM, Depths depths)
	: ht(BytesRAM / sizeof(HashTablePerft::nodetype))
	, depths(depths)
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
	if (depth < depths.cpu_to_gpu + depths.gpu)
		return perft(pos, depth);

	Degeneracies deg(pos, depths.uniquification);
	
	const auto size = static_cast<int64_t>(deg.size());
	std::size_t sum = 0;
	#pragma omp parallel for schedule(dynamic,1) reduction(+:sum)
	for (int64_t i = 0; i < size; i++)
		sum += perft(deg.Position(i), depth - depths.uniquification) * deg.Degeneracy(i);
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
	if (depth <= depths.gpu + depths.cpu_to_gpu && depth > depths.gpu)
		return perft_cpu_to_gpu(pos, depth);

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

uint64_t NumberOfGamesCalculator::perft_cpu_to_gpu(const CPosition& pos, uint8_t depth)
{
	std::vector<CPosition> vec;
	//Children children(pos, depths.cpu_to_gpu, depths.gpu);
	generate_children(pos, depth - depths.gpu, [&vec](CPosition pos) { vec.push_back(pos); });
	return perft_gpu(vec, depths.gpu, depths.blocks, depths.thrads_per_block);
	//auto moves = pos.PossibleMoves();

	//if (moves.empty())
	//{
	//	const auto PosPass = pos.PlayPass();
	//	if (PosPass.HasMoves())
	//		return perft_HT(PosPass, depth - 1);
	//	return 0;
	//}

	//if (const auto ret = ht.LookUp(PerftKey(pos, depth)); ret.has_value())
	//	return ret.value();

	//uint64_t sum = 0;
	//while (!moves.empty())
	//	sum += perft_HT(pos.Play(moves.ExtractMove()), depth - 1);

	//ht.Update(PerftKey(pos, depth), sum);
	//return sum;
}

// perft for 2 plies left
uint64_t NumberOfGamesCalculator::perft_2(const CPosition& pos)
{
	auto moves = pos.PossibleMoves();
	if (moves.empty())
		return pos.PlayPass().PossibleMoves().size();

	uint64_t sum = 0ui64;
	while (!moves.empty())
	{
		const auto pos2 = pos.Play(moves.ExtractMove());
		const auto moves2 = pos2.PossibleMoves();
		if (!moves2.empty())
			sum += moves2.size();
		else
			sum += static_cast<uint64_t>(pos2.PlayPass().HasMoves());
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

NumberOfGamesCalculator::Depths::Depths(uint8_t uniquification, uint8_t cpu_to_gpu, uint8_t gpu, uint16_t blocks, uint16_t thrads_per_block)
	: uniquification(uniquification)
	, cpu_to_gpu(cpu_to_gpu)
	, gpu(gpu)
	, blocks(blocks)
	, thrads_per_block(thrads_per_block)
{}

Children::Children(CPosition pos, uint8_t cpu_depth, uint8_t gpu_depth)
{
	generate_children(pos, cpu_depth - gpu_depth, [this](CPosition pos) { vec.push_back(pos); });
}