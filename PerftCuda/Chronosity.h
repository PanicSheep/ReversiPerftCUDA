#pragma once
#include "cuda_runtime.h"
#include <functional>

inline void ErrorCheck(const std::function<cudaError_t(void)>& fkt) {
	cudaError_t err = fkt();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
}

enum chronosity
{
	syn, // synchron
	asyn // asynchron
};
