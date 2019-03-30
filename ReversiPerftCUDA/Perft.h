#pragma once
#include "Position.h"
#include "HashtablePerft.h"
#include <map>
#include <functional>

void generate_children(CPosition, uint8_t depth, std::function<void(CPosition)>);

class Degeneracies
{
	std::vector<std::pair<CPosition, uint64_t>> vec;
public:
	Degeneracies() = default;
	Degeneracies(CPosition, uint8_t depth);

	std::size_t size() const { return vec.size(); }
	const CPosition& Position(std::size_t index) const { return vec[index].first; }
	std::size_t Degeneracy(std::size_t index) const { return vec[index].second; }
};

class Children
{
	std::vector<CPosition> vec;
public:
	Children(CPosition, uint8_t cpu_depth, uint8_t gpu_depth);

	auto operator[](std::size_t index) const { return vec[index]; }
	auto begin() const { return vec.begin(); }
	auto end() const { return vec.end(); }
};

// Calculates the number of possible Reversi games at the end of the n-th ply.
class NumberOfGamesCalculator
{
	HashTablePerft ht;
	Degeneracies unique_pos_multiplier{};

	struct Depths {
		uint8_t uniques;
		uint8_t uniquification;
		uint8_t cpu_to_gpu;
		uint8_t gpu;

		Depths(uint8_t uniques, uint8_t uniquification, uint8_t cpu_to_gpu, uint8_t gpu);
	} depths;

public:
	NumberOfGamesCalculator(uint64_t BytesRAM, Depths = { 13, 5, 7, 3 });

	static uint64_t CorrectValue(uint8_t depth);

	uint64_t Calc(uint8_t depth);
	uint64_t Calc(const CPosition&, uint8_t depth);

private:
	uint64_t perft           (const CPosition&, uint8_t depth);
	uint64_t perft_HT        (const CPosition&, uint8_t depth);
	uint64_t perft_cpu_to_gpu(const CPosition&, uint8_t depth);
	uint64_t perft_2  (const CPosition&);
	uint64_t perft_1  (const CPosition&);
	uint64_t perft_0  ();
};
