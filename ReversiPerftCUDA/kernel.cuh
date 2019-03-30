#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cstdint>
#include <vector>


void Initialize();

class CPosition;
uint64_t perft_3_gpu(const std::vector<CPosition>&);
