#pragma once
#include "Hashtable.h"
#include <atomic>
#include <cstdint>

struct PerftKey
{
	CPosition pos;
	uint64_t depth;

	PerftKey(const CPosition& pos, uint64_t depth) : pos(pos), depth(depth) {}
};

class BigNode
{
	mutable std::atomic<uint64_t> m_value{ 0 };
	uint64_t m_P{ 0 }, m_O{ 0 }, m_depth{ 0 };
public:
	BigNode() = default;
	BigNode(const BigNode&) = delete;
	BigNode(BigNode&&) = delete;
	BigNode& operator=(const BigNode&) = delete;
	BigNode& operator=(BigNode&&) = delete;

	void Update(const PerftKey& key, const uint64_t value)
	{
		const uint64_t old_value = lock();
		if (value > old_value)
		{
			m_P = key.pos.GetP();
			m_O = key.pos.GetO();
			m_depth = key.depth;
			unlock(value);
		}
		else
			unlock(old_value);
	}

	std::optional<uint64_t> LookUp(const PerftKey& key) const
	{
		const uint64_t old_value = lock();
		const auto P = m_P;
		const auto O = m_O;
		const auto depth = m_depth;
		unlock(old_value);

		if ((key.pos.GetP() == P) && (key.pos.GetO() == O) && (key.depth == depth))
			return old_value;
		return {};
	}

	void Clear()
	{
		lock();
		m_P = 0;
		m_O = 0;
		m_depth = 0;
		unlock(0);
	}

private:
	uint64_t lock() const
	{
		constexpr uint64_t lock_value = 0xFFFFFFFFFFFFFFFui64; // Reserved value to mark node as locked.
		uint64_t value;
		while ((value = m_value.exchange(lock_value, std::memory_order_acquire)) == lock_value)
			continue;
		return value;
	}

	void unlock(uint64_t value) const
	{
		m_value.store(value, std::memory_order_release);
	}
};

static_assert(sizeof(BigNode) <= std::hardware_constructive_interference_size);


struct HashTablePerft : public HashTable<PerftKey, uint64_t, BigNode>
{
	HashTablePerft(uint64_t buckets)
		: HashTable(buckets,
			[](const keytype& key) {
				uint64_t P = key.pos.GetP();
				uint64_t O = key.pos.GetO();
				P ^= P >> 36;
				O ^= O >> 21;
				return (P * O + key.depth);
			})
	{}
};
