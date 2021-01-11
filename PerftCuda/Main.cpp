#include "kernel.cuh"
#include "Core/Core.h"
#include "IO/IO.h"
#include "Perft/Perft.h"
#include "PerftCuda.h"
#include <chrono>
#include <execution>
#include <filesystem>
#include <string>
#include <iostream>
#include <iomanip>
#include <random>
#include <map>

#pragma pack(1)
struct PosResDur
{
	Position pos;
	uint64 value;
	double duration;

	bool operator==(const PosResDur& o) const { return pos == o.pos; }
	bool operator<(const PosResDur& o) const { return pos < o.pos; }
};
#pragma pack()

std::vector<PosResDur> ReadFile(const std::filesystem::path& path)
{
	std::vector<PosResDur> content;
	std::fstream stream(path, std::ios::in | std::ios::binary);
	auto eof = [&](){ stream.peek(); return stream.eof(); };
	while (stream && not eof())
	{
		PosResDur buffer;
		stream.read(reinterpret_cast<char*>(&buffer), sizeof(PosResDur));
		content.push_back(buffer);
	}
	stream.close();
	return content;
}

void PrintHelp()
{
	std::cout
		<< "This program calculates the number of possible Reversi games at the end of the n-th ply.\n"
		<< "In correspondence with (https://oeis.org/A124004).\n\n"
		<< "Args:\n"
		<< " -f 'string'      File path.\n"
		<< " -fd 'int'        Ply depth of positions in file.\n"
		<< " -d 'int'         Ply depth to evaluate positions in file.\n"
		<< " -ram 'string'    Size of the hash table in bytes (default: '1GB').\n"
		<< " -t 'int' 'int'   Tests the results of position number 'int' to 'int', not including the latter.\n"
		<< " -ta              Test all.\n"
		<< " -cuda            Run on cuda capable GPUs.\n"
		<< " -h               Prints this help.\n\n"
		<< "Possible ways to use this program:\n"
		<< " -f 'string' -fd 'int' -d 'int' (-cuda) (-ram 'int')\n"
		<< "   Creates a file, if it does not exist, containing all possible Reversi positions as the end of the (fd)-th ply.\n"
		<< "   It will open the file and calculate the number of possible Reversi games at the end of each position's (d)-th ply.\n"
		<< "   When done it will sum up the intermediate results to conclude the number of possible Reversi positions at the end of the (fd+d)-th ply.\n"
		<< " -f 'string' -fd 'int' -d 'int' -v 'int' 'int (-cuda) (-ram 'int')\n"
		<< "   Tests the intermediate results in the file for correctness."
		<< " -f 'string' -fd 'int' -d 'int' -va (-cuda) (-ram 'int')\n"
		<< "   Tests all intermediate results for correctness."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	std::filesystem::path file;
	uint file_depth, depth;
	uint64 RAM{0}, test_begin{0}, test_end{0};
	bool cuda = false;
	bool test_all = false;
	bool test = false;

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-f") file = argv[++i];
		else if (std::string(argv[i]) == "-fd") file_depth = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-d") depth = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-ram") RAM = ParseBytes(argv[++i]);
		else if (std::string(argv[i]) == "-t") { test_begin = std::stoi(argv[++i]); test_end = std::stoi(argv[++i]); test = true; }
		else if (std::string(argv[i]) == "-ta") { test_all = true; test = true; }
		else if (std::string(argv[i]) == "-cuda") cuda = true;
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	std::locale locale("");
	std::cout.imbue(locale);
	std::cout << std::setfill(' ') << std::boolalpha;
	std::cout << "Running with:\n"
		<< "File: " << file << "\n"
		<< "File depth: " << file_depth << "\n"
		<< "Depth: " << depth << "\n"
		<< "RAM: " << RAM << "\n"
		<< "Testing: " << test << "\n"
		<< "Testing all: " << test_all << "\n"
		<< "Range: " << test_begin << " to " << test_end << "\n"
		<< "CUDA: " << cuda << std::endl;

	std::unique_ptr<BasicPerft> engine;
	if (cuda && RAM)
		engine = std::make_unique<CudaHashTablePerft>(RAM, 3, 5, 3);
	else if (RAM)
		engine = std::make_unique<HashTablePerft>(RAM, 6);
	else
		engine = std::make_unique<UnrolledPerft>(6);

	std::vector<PosResDur> work = ReadFile(file);
	std::cout << "Positions in file: " << work.size() << "\n\n";
	if (test_all)
		test_end = work.size();

	if (work.empty()) // file does not exist
	{
		// create all unique positions at the end of the 'file_depth'-th ply.
		std::fstream stream(file, std::ios::out | std::ios::binary);
		for (const auto& pos : Children(Position::Start(), file_depth, true))
			work.push_back({FlipToUnique(pos), 0, 0.0});
		std::sort(std::execution::par_unseq, work.begin(), work.end());
		auto last = std::unique(std::execution::par_unseq, work.begin(), work.end());
		work.erase(last, work.end());
		std::shuffle(work.begin(), work.end(), std::default_random_engine{});
	}

	if (test)
	{
		std::cout << "   # |       Positions       |correct|      Time[s]     |       Pos/s      | ETA \n";
		std::cout << "-----+-----------------------+-------+------------------+------------------+-----\n";

		double total_duration = 0.0;
		for (std::size_t i = test_begin; i < test_end; i++)
		{
			const auto start = std::chrono::high_resolution_clock::now();
			const auto value = engine->calculate(work[i].pos, depth - file_depth);
			const auto stop = std::chrono::high_resolution_clock::now();
			work[i].duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1'000.0;

			total_duration += work[i].duration;
			std::cout << std::setw(4) << i << " |";
			std::cout << std::setw(22) << work[i].value << " |";
			std::cout << std::setw(6) << (value == work[i].value) << " |";
			std::cout << std::setw(17) << work[i].duration << " | ";
			if (work[i].duration)
				std::cout << std::setw(17) << int64(work[i].value / work[i].duration)  << " | ";
			auto t = std::chrono::system_clock::to_time_t(
				std::chrono::system_clock::now()
				+ std::chrono::seconds(static_cast<long long>(
					total_duration / (i - test_begin + 1) * (test_end - i - 1)
					)));
			std::cout << std::ctime(&t);

			WriteToFile(file, work);
		}
	}
	else
	{
		std::cout << "   # |       Positions       |      Time[s]     |       Pos/s      | ETA \n";
		std::cout << "-----+-----------------------+------------------+------------------+-----\n";

		double total_duration = 0.0;
		for (std::size_t i = 0; i < work.size(); i++)
		{
			if (work[i].value == 0)
			{
				const auto start = std::chrono::high_resolution_clock::now();
				work[i].value = engine->calculate(work[i].pos, depth - file_depth);
				const auto stop = std::chrono::high_resolution_clock::now();
				work[i].duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1'000.0;
			}
			total_duration += work[i].duration;
			std::cout << std::setw(4) << i << " |";
			std::cout << std::setw(22) << work[i].value << " |";
			std::cout << std::setw(17) << work[i].duration << " | ";
			if (work[i].duration)
				std::cout << std::setw(17) << int64(work[i].value / work[i].duration)  << " | ";
			auto t = std::chrono::system_clock::to_time_t(
				std::chrono::system_clock::now()
				+ std::chrono::seconds(static_cast<long long>(
					total_duration / (i + 1) * (work.size() - i - 1)
					)));
			std::cout << std::ctime(&t);

			WriteToFile(file, work);
		}

		std::map<Position, uint64> value;
		for (const auto& w : work)
			value[w.pos] = w.value;

		// calculate final result
		uint64 sum = 0;
		for (const auto& pos : Children(Position::Start(), file_depth, true))
			sum += value[FlipToUnique(pos)];
		std::cout << "Sum: " << sum << std::endl;
		std::cout << "Duration: " << total_duration << std::endl;
	}


	if (cuda)
	{
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}
    return 0;
}
