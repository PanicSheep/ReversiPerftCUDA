#pragma once
#include "DeviceVector.cuh"
#include "HostVector.cuh"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cstdint>
#include <vector>

void Initialize();

class CPosition;

uint64_t perft_gpu(const std::vector<CPosition>&, int gpu_depth, int blocks, int threads_per_block);