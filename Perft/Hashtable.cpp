#include "Hashtable.h"

void BigNode::Update(const key_type& new_key, const value_type& new_value)
{
	ScopedLock<decltype(m_value)> lock_guard{m_value};
	if (new_value > lock_guard.value)
	{
		m_key = new_key;
		lock_guard.value = new_value;
	}
}

std::optional<BigNode::value_type> BigNode::LookUp(const key_type& key) const
{
	const auto pair = Get();
	if (pair.key == key)
		return pair.value;
	return {};
}

void BigNode::Clear()
{
	ScopedLock<decltype(m_value)> lock_guard{m_value};
	m_key = key_type{};
	lock_guard.value = 0;
}

BigNode::KeyValuePair BigNode::Get() const
{
	ScopedLock<decltype(m_value)> lock_guard{m_value};
	return { m_key, lock_guard.value };
}
