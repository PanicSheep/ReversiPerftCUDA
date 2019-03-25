#pragma once
#include "Position.h"
#include "HashtablePerft.h"
#include <map>

class DegeneracyCounter
{
	std::map<CPosition, std::size_t> map;
public:
	void Insert(CPosition pos) { map[std::move(pos)]++; }
	auto begin() const { return map.begin(); }
	auto end() const { return map.end(); }
};

class DegeneracyAccessor
{
	std::vector<std::pair<CPosition, std::size_t>> vec;
public:
	DegeneracyAccessor(const DegeneracyCounter& o) : vec(o.begin(), o.end()) {}
	std::size_t size() const { return vec.size(); }
	const CPosition& Position(std::size_t index) const { return vec[index].first; }
	std::size_t Degeneracy(std::size_t index) const { return vec[index].second; }
};

// Calculates the number of possible Reversi games at the end of the n-th ply.
class NumberOfGamesCalculator
{
	HashTablePerft ht;
	DegeneracyCounter unique_pos_multiplier;
	struct HyperParameter {
		uint8_t uniqueness_optimization_depth;
		std::size_t initial_depth;
		uint8_t gpu;
		HyperParameter(uint8_t uniqueness_optimization_depth, std::size_t initial_depth, uint8_t gpu);
	} hyperparameter;

public:
	NumberOfGamesCalculator(uint64_t BytesRAM, HyperParameter = { 13, 9, 4 });

	static uint64_t CorrectValue(uint8_t depth);

	uint64_t Calc(uint8_t depth);
	uint64_t Calc(const CPosition&, uint8_t depth);

private:
	uint64_t perft   (const CPosition&, uint8_t depth);
	uint64_t perft_HT(const CPosition&, uint8_t depth);
	uint64_t perft_d (const CPosition&, uint8_t depth);
	uint64_t perft_2 (const CPosition&);
	uint64_t perft_1 (const CPosition&);
	uint64_t perft_0 ();

	void fill_unique_pos_multiplier(CPosition, uint8_t depth);
};
