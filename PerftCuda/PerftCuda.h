#pragma once
#include "Perft/Perft.h"

class CudaHashTablePerft : public HashTablePerft
{
	const int pre_cuda_unroll, cuda_depth;
protected:
	int64 calculate_n(const Position&, int depth);
public:
	CudaHashTablePerft(std::size_t bytes, int initial_unroll, int pre_cuda_unroll, int cuda_depth)
		: HashTablePerft(bytes, initial_unroll), pre_cuda_unroll(pre_cuda_unroll), cuda_depth(cuda_depth) {}

	int64 calculate(const Position&, int depth) override;
	int64 calculate(int depth) override;
};