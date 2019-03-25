#pragma once
#include "Position.h"
#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <functional>

template <typename KeyType, typename ValueType>
struct IHashTable
{
	virtual ~IHashTable() = default;

	virtual void Update(const KeyType&, const ValueType&) = 0;
	virtual std::optional<ValueType> LookUp(const KeyType&) const = 0;
	virtual void Clear() = 0;
};

template <typename KeyType, typename ValueType, typename NodeType>
class HashTable : public IHashTable<KeyType, ValueType>
{
	mutable std::atomic<uint64_t> update_counter{ 0 }, lookup_counter{ 0 }, hit_counter{ 0 };
	std::function<std::size_t(const KeyType&)> hash;
	std::vector<NodeType> table;
	
public:
	using nodetype = NodeType;
	using keytype = KeyType;
	using valuetype = ValueType;

	HashTable(uint64_t buckets, std::function<std::size_t(const KeyType&)> hash = [](const KeyType& key) { return std::hash<KeyType>()(key); })
		: table(buckets)
		, hash(std::move(hash)){}
	
	void Update(const KeyType&, const ValueType&) override;
	std::optional<ValueType> LookUp(const KeyType&) const override;
	void Clear() override;
	void PrintStatistics();
};

template <typename KeyType, typename ValueType, typename NodeType>
inline void HashTable<KeyType, ValueType, NodeType>::Update(const KeyType& key, const ValueType& value)
{
	update_counter++;
	table[hash(key) % table.size()].Update(key, value);
}

template <typename KeyType, typename ValueType, typename NodeType>
inline std::optional<ValueType> HashTable<KeyType, ValueType, NodeType>::LookUp(const KeyType& key) const
{
	lookup_counter++;
	const auto ret = table[hash(key) % table.size()].LookUp(key);
	if (ret.has_value())
		hit_counter++;
	return ret;
}

template <typename KeyType, typename ValueType, typename NodeType>
inline void HashTable<KeyType, ValueType, NodeType>::Clear()
{
	update_counter = 0;
	lookup_counter = 0;
	hit_counter = 0;
	for (auto& it : table)
		it.Clear();
}

template <typename KeyType, typename ValueType, typename NodeType>
inline void HashTable<KeyType, ValueType, NodeType>::PrintStatistics()
{
	uint64_t counter[3] = { 0,0,0 };
	for (const auto& it : table)
		counter[it.NumberOfNonEmptyNodes()]++;
	float total = static_cast<float>(counter[0] + counter[1] + counter[2]);

	std::cout
		<< "Size:     " << ThousandsSeparator(sizeof(NodeType)) << " Bytes\n"
		<< "Updates:  " << ThousandsSeparator(update_counter) << "\n"
		<< "LookUps:  " << ThousandsSeparator(lookup_counter) << "\n"
		<< "Hits:     " << ThousandsSeparator(hit_counter) << "\n"
		<< "Hit rate: " << static_cast<float>(hit_counter * 100) / static_cast<float>(lookup_counter) << "%\n"
		<< "Zero entry nodes : " << counter[0] / total * 100 << "%\n"
		<< "One  entry nodes : " << counter[1] / total * 100 << "%\n"
		<< "Two  entry nodes : " << counter[2] / total * 100 << "%\n";
}