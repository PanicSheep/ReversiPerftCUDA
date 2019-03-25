#pragma once
#include "MacrosHell.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <ratio>
#include <sstream>
#include <string>
#include <vector>

#ifdef _MSC_VER
	#include <direct.h>
#else
	#include <unistd.h>
#endif

void replace_all(std::string& source, const std::string& from, const std::string& to);

std::vector<std::string> split(const std::string& source, const std::string& delimitter);
std::string join(const std::vector<std::string>& parts, const std::string& delimitter);

uint32_t Pow_int(uint32_t base, uint32_t exponent);


inline int RoundInt(const double d) { return static_cast<int>(std::round(d)); } // TODO: Remove?
inline int RoundInt(const float  f) { return static_cast<int>(std::round(f)); } // TODO: Remove?

std::string time_format(std::chrono::milliseconds duration);

template <class U, class V>
std::string time_format(std::chrono::duration<U, V> duration)
{
	return time_format(std::chrono::duration_cast<std::chrono::milliseconds>(duration));
}

std::string short_time_format(std::chrono::duration<long long, std::pico> duration);
std::string ThousandsSeparator(uint64_t n);
std::string DateTimeNow();

// Returns an int with either a '+' or a '-' prefix. Zero is represented '+0'.
std::string SignedInt(int score);

// Returns a double digit int with either a '+' or a '-' prefix. Zero is represented '+00'.
 std::string DoubleDigitSignedInt(int score);

 std::size_t ParseBytes(const std::string& bytes);

uint64_t FlipCodiagonal(uint64_t);
uint64_t FlipDiagonal(uint64_t);
uint64_t FlipHorizontal(uint64_t);
uint64_t FlipVertical(uint64_t);
