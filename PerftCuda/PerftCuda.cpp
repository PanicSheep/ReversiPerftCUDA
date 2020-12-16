#include "PerftCuda.h"
#include "Core/Core.h"

// forward declaration
int64 perft_cuda(const Position&, const int depth, const int cuda_depth);

int64 CudaHashTablePerft::calculate_n(const Position& pos, const int depth)
{
	if (depth <= 3)
		return UnrolledPerft::calculate(pos, depth);
	if (depth <= pre_cuda_unroll + cuda_depth)
		return perft_cuda(pos, depth, cuda_depth);

	if (auto ret = hash_table.LookUp({ pos, depth }); ret.has_value())
		return ret.value();

	auto moves = PossibleMoves(pos);
	if (!moves)
	{
		Position passed = PlayPass(pos);
		if (HasMoves(passed))
			return calculate_n(FlipToUnique(passed), depth-1);
		return 0;
	}

	int64 sum = 0;
	for (const auto& move : moves)
		sum += calculate_n(FlipToUnique(Play(pos, move)), depth-1);

	hash_table.Update({ pos, depth }, sum);
	return sum;
}

int64 CudaHashTablePerft::calculate(const Position& pos, const int depth)
{
	if (initial_unroll == 0 || depth - initial_unroll <= 2)
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

int64 CudaHashTablePerft::calculate(const int depth)
{
	if (depth == 0)
		return calculate(Position::Start(), depth);

	// Makes use of 4-fold symmetrie.
	Position pos = Position::Start();
	pos = Play(pos, PossibleMoves(pos).First());
	return 4 * calculate(pos, depth-1);
}
