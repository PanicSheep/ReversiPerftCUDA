#pragma once
#include <type_traits>
#include <functional>
#include <vector>
#include <numeric>

template <class T>
inline std::vector<T> range(T begin, T end)
{
	std::vector<T> elements(end - begin);
	std::iota(elements.begin(), elements.end(), begin);
	return elements;
}
