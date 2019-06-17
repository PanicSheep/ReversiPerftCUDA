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

constexpr uint64_t operator""_kB(uint64_t kilo_byte) { return kilo_byte * 1024; }
constexpr uint64_t operator""_MB(uint64_t mega_byte) { return mega_byte * 1024 * 1024; }
constexpr uint64_t operator""_GB(uint64_t giga_byte) { return giga_byte * 1024 * 1024 * 1024; }

int main(int argc, char* argv[])
{
	Initialize();
	unsigned int depth = 21;
	constexpr std::size_t RAM = 1_GB;

	//for (int i = 0; i < argc; i++)
	//{
	//	if (std::string(argv[i]) == "-d") depth = atoi(argv[++i]);
	//	else if (std::string(argv[i]) == "-RAM") RAM = ParseBytes(argv[++i]);
	//	else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	//}

	std::cout << "depth|        Positions         |correct|       Time       |       N/s       " << std::endl;
	std::cout << "-----+--------------------------+-------+------------------+-----------------" << std::endl;

	for (uint8_t d = 1; d <= 16; d++)
	{
		auto startTime = std::chrono::high_resolution_clock::now();
		auto result = NumberOfGamesCalculator{ RAM, { /*uniquification*/ 6, /*cpu_to_gpu*/ 7, /*gpu*/ 4, /*blocks*/ 512, /*thrads_per_block*/ 128 } }.Calc(d);
		auto endTime = std::chrono::high_resolution_clock::now();
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
