#include "kernel.cuh"
#include "MacrosHell.h"
#include "Position.h"
#include <numeric>
#include <omp.h>

__constant__ uint8_t  d_OUTFLANK[8][ 64];
__constant__ uint8_t  d_FLIPS   [8][256];
__constant__ uint64_t d_STRETCH    [256];
__constant__ uint64_t d_MASK_D     [ 64];
__constant__ uint64_t d_MASK_C     [ 64];

void Initialize()
{
	uint8_t OUTFLANK[8][64];
	uint8_t FLIPS[8][256];
	uint64_t STRETCH[256];
	uint64_t MASK_D[64];
	uint64_t MASK_C[64];

	uint8_t O, k, outf;
	for (uint8_t j = 0; j < 8; j++)
	{
		for (uint8_t i = 0; i < 64; i++)
		{
			outf = 0;
			O = i << 1;

			k = j + 1;
			while (O & (1 << k))
				k++;
			if (k != j + 1) // There was an outflank
				outf |= 1 << k;

			k = j - 1;
			while (O & (1 << k))
				k--;
			if (k != j - 1) // There was an outflank
				outf |= 1 << k;

			OUTFLANK[j][i] = outf;
		}
	}

	for (unsigned int i = 0; i < 8; i++)
		for (unsigned int j = 0; j < 256; j++)
			FLIPS[i][j] = 0;

	// 0000 000X
	FLIPS[0][0x04] = 0x02;
	FLIPS[0][0x08] = 0x06;
	FLIPS[0][0x10] = 0x0E;
	FLIPS[0][0x20] = 0x1E;
	FLIPS[0][0x40] = 0x3E;
	FLIPS[0][0x80] = 0x7E;

	// 0000 00X0
	FLIPS[1][0x08] = 0x04;
	FLIPS[1][0x10] = 0x0C;
	FLIPS[1][0x20] = 0x1C;
	FLIPS[1][0x40] = 0x3C;
	FLIPS[1][0x80] = 0x7C;

	// 0000 0X00
	FLIPS[2][0x01] = 0x02;
	FLIPS[2][0x10] = 0x08;
	FLIPS[2][0x11] = 0x0A;
	FLIPS[2][0x20] = 0x18;
	FLIPS[2][0x21] = 0x1A;
	FLIPS[2][0x40] = 0x38;
	FLIPS[2][0x41] = 0x3A;
	FLIPS[2][0x80] = 0x78;
	FLIPS[2][0x81] = 0x7A;

	// 0000 X000
	FLIPS[3][0x01] = 0x06;
	FLIPS[3][0x02] = 0x04;
	FLIPS[3][0x20] = 0x10;
	FLIPS[3][0x21] = 0x16;
	FLIPS[3][0x22] = 0x14;
	FLIPS[3][0x40] = 0x30;
	FLIPS[3][0x41] = 0x36;
	FLIPS[3][0x42] = 0x34;
	FLIPS[3][0x80] = 0x70;
	FLIPS[3][0x81] = 0x76;
	FLIPS[3][0x82] = 0x74;

	// 000X 0000
	FLIPS[4][0x01] = 0x0E;
	FLIPS[4][0x02] = 0x0C;
	FLIPS[4][0x04] = 0x08;
	FLIPS[4][0x40] = 0x20;
	FLIPS[4][0x41] = 0x2E;
	FLIPS[4][0x42] = 0x2C;
	FLIPS[4][0x44] = 0x28;
	FLIPS[4][0x80] = 0x60;
	FLIPS[4][0x81] = 0x6E;
	FLIPS[4][0x82] = 0x6C;
	FLIPS[4][0x84] = 0x68;

	// 00X0 0000
	FLIPS[5][0x01] = 0x1E;
	FLIPS[5][0x02] = 0x1C;
	FLIPS[5][0x04] = 0x18;
	FLIPS[5][0x08] = 0x10;
	FLIPS[5][0x80] = 0x40;
	FLIPS[5][0x81] = 0x5E;
	FLIPS[5][0x82] = 0x5C;
	FLIPS[5][0x84] = 0x58;
	FLIPS[5][0x88] = 0x50;

	// 0X00 0000
	FLIPS[6][0x01] = 0x3E;
	FLIPS[6][0x02] = 0x3C;
	FLIPS[6][0x04] = 0x38;
	FLIPS[6][0x08] = 0x30;
	FLIPS[6][0x10] = 0x20;

	// X000 0000
	FLIPS[7][0x01] = 0x7E;
	FLIPS[7][0x02] = 0x7C;
	FLIPS[7][0x04] = 0x78;
	FLIPS[7][0x08] = 0x70;
	FLIPS[7][0x10] = 0x60;
	FLIPS[7][0x20] = 0x40;

	for (uint64_t i = 0; i < 256; i++)
		STRETCH[i] = ((i * 0x0102040810204080ULL) & 0x0101010101010101ULL) * 0xFFULL;

	for (unsigned int i = 0; i < 64; i++)
	{
		unsigned int L = i % 8;
		unsigned int N = i / 8;
		if (L > N) MASK_D[i] = 0x8040201008040201ULL >> ((L - N) * 8);
		else MASK_D[i] = 0x8040201008040201ULL << ((N - L) * 8);
	}

	for (unsigned int i = 0; i < 64; i++)
	{
		unsigned int L = i % 8;
		unsigned int N = i / 8;
		if (N + L > 7) MASK_C[i] = 0x0102040810204080ULL << ((N + L - 7) * 8);
		else MASK_C[i] = 0x0102040810204080ULL >> (-(N + L - 7) * 8);
	}

	for (int i = 0; i < 2; i++)
	{
		cudaSetDevice(i);
		cudaMemcpyToSymbol(d_OUTFLANK, OUTFLANK, sizeof(uint8_t) * 8 * 64);
		cudaMemcpyToSymbol(d_FLIPS, FLIPS, sizeof(uint8_t) * 8 * 256);
		cudaMemcpyToSymbol(d_STRETCH, STRETCH, sizeof(uint64_t) * 256);
		cudaMemcpyToSymbol(d_MASK_D, MASK_D, sizeof(uint64_t) * 64);
		cudaMemcpyToSymbol(d_MASK_C, MASK_C, sizeof(uint64_t) * 64);
	}
}

template <const unsigned int dir>
__device__ __inline__ uint64_t CUDA_get_some_moves(const uint64_t P, const uint64_t mask)
{
	// kogge-stone parallel prefix
	// 12 x SHIFT, 10 x AND, 7 x OR
	// = 29 OPs
	uint64_t flip_l, flip_r;
	uint64_t mask_l, mask_r;

	flip_l = mask & (P << dir);
	flip_r = mask & (P >> dir);

	flip_l |= mask & (flip_l << dir);
	flip_r |= mask & (flip_r >> dir);

	mask_l = mask & (mask << dir);
	mask_r = mask & (mask >> dir);

	flip_l |= mask_l & (flip_l << (dir * 2));
	flip_r |= mask_r & (flip_r >> (dir * 2));

	flip_l |= mask_l & (flip_l << (dir * 2));
	flip_r |= mask_r & (flip_r >> (dir * 2));

	return (flip_l << dir) | (flip_r >> dir);
}

__device__ uint64_t CUDA_HasMoves(const CPosition& pos)
{
	const uint64_t empties = pos.Empties();
	if (CUDA_get_some_moves<1>(pos.GetP(), pos.GetO() & 0x7E7E7E7E7E7E7E7EULL) & empties) return 1;
	if (CUDA_get_some_moves<8>(pos.GetP(), pos.GetO() & 0x00FFFFFFFFFFFF00ULL) & empties) return 1;
	if (CUDA_get_some_moves<7>(pos.GetP(), pos.GetO() & 0x007E7E7E7E7E7E00ULL) & empties) return 1;
	if (CUDA_get_some_moves<9>(pos.GetP(), pos.GetO() & 0x007E7E7E7E7E7E00ULL) & empties) return 1;
	return 0;
}

__device__ __inline__ uint64_t CUDA_flip_h(const CPosition& pos, const uint8_t move)
{
	const uint64_t O = (pos.GetO() >> ((move & 0xF8) + 1)) & 0x3FULL;
	const uint64_t P = (pos.GetP() >> (move & 0xF8)) & 0xFFULL;
	const uint64_t outflank = d_OUTFLANK[move & 7][O] & P;
	return static_cast<uint64_t>(d_FLIPS[move & 7][outflank]) << (move & 0xF8);
}

__device__ __inline__ uint64_t CUDA_flip_v(const CPosition& pos, const uint8_t move)
{
	const uint64_t O = ((pos.GetO() & (0x0001010101010100ULL << (move & 7))) * (0x0102040810204080ULL >> (move & 7))) >> 57;
	const uint64_t P = ((pos.GetP() & (0x0101010101010101ULL << (move & 7))) * (0x0102040810204080ULL >> (move & 7))) >> 56;
	const uint64_t outflank = d_OUTFLANK[(move >> 3)][O] & P;
	return d_STRETCH[d_FLIPS[(move >> 3)][outflank]] & (0x0101010101010101ULL << (move & 7));
}

__device__ __inline__ uint64_t CUDA_flip_d(const CPosition& pos, const uint8_t move)
{
	const uint64_t O = ((pos.GetO() & d_MASK_D[move] & 0x007E7E7E7E7E7E00ULL) * 0x0101010101010101ULL) >> 57;
	const uint64_t P = ((pos.GetP() & d_MASK_D[move]) * 0x0101010101010101ULL) >> 56;
	const uint64_t outflank = d_OUTFLANK[move & 7][O] & P;
	return (d_FLIPS[move & 7][outflank] * 0x0101010101010101ULL) & d_MASK_D[move];
}

__device__ __inline__ uint64_t CUDA_flip_c(const CPosition& pos, const uint8_t move)
{
	const uint64_t O = ((pos.GetO() & d_MASK_C[move] & 0x007E7E7E7E7E7E00ULL) * 0x0101010101010101ULL) >> 57;
	const uint64_t P = ((pos.GetP() & d_MASK_C[move]) * 0x0101010101010101ULL) >> 56;
	const uint64_t outflank = d_OUTFLANK[move & 7][O] & P;
	return (d_FLIPS[move & 7][outflank] * 0x0101010101010101ULL) & d_MASK_C[move];
}

__device__ uint64_t CUDA_flip(const CPosition& pos, const uint8_t move)
{
	const auto h = CUDA_flip_h(pos, move);
	const auto v = CUDA_flip_v(pos, move);
	const auto d = CUDA_flip_d(pos, move);
	const auto c = CUDA_flip_c(pos, move);
	return h | v | d | c;
}

template <int depth>
__device__ uint64_t GPUperft(const CPosition& pos)
{
	auto moves = PossibleMoves(pos);
	if (moves.empty())
		return GPUperft<depth - 1>(pos.PlayPass());

	uint64_t sum = 0;
	while (!moves.empty())
	{
		auto move = moves.ExtractMove();
		uint64_t flipped = CUDA_flip(pos, move);
		sum += GPUperft<depth - 1>(pos.Play(move, flipped));
	}
	return sum;
}

template<>
__device__ uint64_t GPUperft<2>(const CPosition& pos)
{
	auto moves = PossibleMoves(pos);
	if (moves.empty())
		return PossibleMoves(pos.PlayPass()).size();

	uint64_t sum = 0;
	while (!moves.empty())
	{
		auto move = moves.ExtractMove();
		uint64_t flipped = CUDA_flip(pos, move);
		const auto next_pos = pos.Play(move, flipped);
		auto next_moves = PossibleMoves(next_pos);
		if (next_moves.empty())
			sum += CUDA_HasMoves(next_pos.PlayPass());
		else
			sum += next_moves.size();
	}
	return sum;
}

template <int depth>
__global__ void kernel(const DeviceVector<CPosition> pos, DeviceVector<uint64_t> result)
{
	const std::size_t tid = threadIdx.x;
	const std::size_t gridSize = blockDim.x * gridDim.x;

	for (int i = tid + blockIdx.x * blockDim.x; i < pos.size(); i += gridSize)
		result[i] = GPUperft<depth>(pos[i]);
}

//template <int depth>
//__device__ void GenerateNexts(const CPosition& pos, CPosition*& out)
//{
//	auto moves = PossibleMoves(in);
//	if (moves.empty())
//		GenerateNexts<depth - 1>(pos.PlayPass(), out);
//
//	while (!moves.empty())
//	{
//		auto move = moves.ExtractMove();
//		uint64_t flipped = CUDA_flip(pos, move);
//		GenerateNexts<depth - 1>(pos.Play(move, flipped), out);
//	}
//}
//
//template<>
//__device__ void GenerateNexts<0>(const CPosition& pos, CPosition*& out)
//{
//	auto moves = PossibleMoves(pos);
//	if (moves.empty())
//		*(out++) = pos.PlayPass();
//
//	while (!moves.empty())
//	{
//		auto move = moves.ExtractMove();
//		uint64_t flipped = CUDA_flip(pos, move);
//		*(out++) = pos.Play(move, flipped);
//	}
//}

//template <int depth>
//__global__ uint64_t kernel2(const CPosition * in, CPosition ** mid, uint64_t size)
//{
//	const std::size_t tid = threadIdx.x;
//	const std::size_t gridSize = blockDim.x * gridDim.x;
//
//	for (int i = tid + blockIdx.x * blockDim.x; i < size; i += gridSize)
//	{
//		GenerateNexts<depth>(pos[i], mid[i * ??? ]);
//		mid[i] = GPUperft<depth>(pos[i]);
//	}
//}

uint64_t perft_gpu(const std::vector<CPosition>& pos, int gpu_depth, int blocks, int threads_per_block)
{
	static thread_local int tid = []() { int n; cudaGetDeviceCount(&n); int tid = omp_get_thread_num() % n; cudaSetDevice(tid); return tid; }();
	static thread_local HostVector<CPosition> host_pos(1'000'000);
	static thread_local HostVector<uint64_t> host_res(1'000'000);
	static thread_local DeviceVector<CPosition> device_pos(1'000'000);
	static thread_local DeviceVector<uint64_t> device_res(1'000'000);

	host_pos = pos;
	device_pos.store(host_pos, asyn);

	device_res.resize(pos.size());

	switch (gpu_depth)
	{
	case 2: kernel<2><<<blocks, threads_per_block>>>(device_pos, device_res); break;
	case 3: kernel<3><<<blocks, threads_per_block>>>(device_pos, device_res); break;
	case 4: kernel<4><<<blocks, threads_per_block>>>(device_pos, device_res); break;
	case 5: kernel<5><<<blocks, threads_per_block>>>(device_pos, device_res); break;
	}
	cudaError_t err = cudaDeviceSynchronize();
	if (err != 0) {
		auto tmp = cudaGetErrorString(err);
		std::cerr << tmp << std::endl;
	}
	host_res = device_res;
	
	return std::accumulate(host_res.begin(), host_res.end(), 0ui64);
}
