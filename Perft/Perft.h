#pragma once
#include "Core/Position.h"
#include "Hashtable.h"

int64 Correct(int depth);

class BasicPerft
{
public:
	virtual int64 calculate(const Position&, int depth);
	virtual int64 calculate(int depth);

	virtual void clear() {};
};

class UnrolledPerft : public BasicPerft
{
protected:
	const int initial_unroll;
private:
	int64 calculate_0();
	int64 calculate_1(const Position&);
	int64 calculate_2(const Position&);
protected:
	int64 calculate_n(const Position&, int depth);
public:
	UnrolledPerft(int initial_unroll) : initial_unroll(initial_unroll) {}

	int64 calculate(const Position&, int depth) override;
	int64 calculate(int depth) override;
};

class HashTablePerft : public UnrolledPerft
{
protected:
	BigNodeHashTable hash_table;

	int64 calculate_n(const Position&, int depth);
public:
	HashTablePerft(std::size_t bytes, int initial_unroll);

	int64 calculate(const Position&, int depth) override;
	int64 calculate(int depth) override;

	void clear() override { hash_table.Clear(); }
};
