#pragma once
#include <atomic>
#include <functional>
#include <vector>
#include <cstdint>
#include <optional>
#include "Bit.h" // TODO: Replace with MacrosHell!

template <typename Key, typename Value, typename Node>
class HashTable
{
	mutable std::atomic<std::size_t> updates{ 0 }, lookups{ 0 }, hits{ 0 };
	std::function<std::size_t(const Key&)> hash;
	std::vector<Node> buckets;

public:
	using key_type = Key;
	using value_type = Value;
	using node_type = Node;

	HashTable(std::size_t bucket_count, std::function<std::size_t(const Key&)> hash)
		: hash(std::move(hash))
		, buckets(bucket_count)
	{}

	void Update(const Key& key, const Value& value)
	{
		updates++;
		auto index = hash(key) % buckets.size();
		buckets[index].Update(key, value);
	}

	std::optional<Value> LookUp(const Key& key) const
	{
		lookups++;
		auto index = hash(key) % buckets.size();
		const auto ret = buckets[index].LookUp(key);
		if (ret.has_value())
			++hits;
		return ret;
	}

	void Clear()
	{
		updates = 0;
		lookups = 0;
		hits = 0;
		for (auto& bucket : buckets)
			bucket.Clear();
	}

	std::size_t Buckets() const { return buckets.size(); }
	std::size_t MemoryFootprint() const { return buckets.size() * sizeof(Node); }
	std::size_t UpdateCounter() const { return updates; }
	std::size_t LookUpCounter() const { return lookups; }
	std::size_t HitCounter() const { return hits; }
};