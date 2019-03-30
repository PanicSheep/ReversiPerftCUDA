#include "Perft.h"
#include "Utility.h"
#include "kernel.cuh"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <omp.h>
#include <random>
#include <thread>

void PrintHelp()
{
	// TODO
	//std::cout
	//	<< "   -d    Depth of perft.\n"
	//	<< "   -RAM  Number of hash table bytes.\n"
	//	<< "   -h    Prints this help."
	//	<< std::endl;
}

int main(int argc, char* argv[])
{
	Initialize();
	unsigned int depth = 21;
	std::size_t RAM = 1024 * 1024 * 1024;

	//for (int i = 0; i < argc; i++)
	//{
	//	if (std::string(argv[i]) == "-d") depth = atoi(argv[++i]);
	//	else if (std::string(argv[i]) == "-RAM") RAM = ParseBytes(argv[++i]);
	//	else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	//}

	std::cout << "depth|        Positions         |correct|       Time       |       N/s       " << std::endl;
	std::cout << "-----+--------------------------+-------+------------------+-----------------" << std::endl;

	std::chrono::high_resolution_clock::time_point startTime, endTime;
	for (uint8_t d = 1; d <= depth; d++)
	{
		startTime = std::chrono::high_resolution_clock::now();
		uint64_t result = NumberOfGamesCalculator{RAM}.Calc(d);
		endTime = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

		printf(" %3u | %24s |%7s| %14s | %15s\n",
			d,
			ThousandsSeparator(result).c_str(), (NumberOfGamesCalculator::CorrectValue(d) == result ? "  true " : " false "),
			time_format(endTime - startTime).c_str(),
			ms > 0 ? ThousandsSeparator(result / ms * 1000).c_str() : ""
		);
	}

	return 0;
}
