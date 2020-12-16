#pragma once
#include "Core/Core.h"
#include <string>

// Maps input to (.., "-1", "+0", "+1", ..)
std::wstring SignedInt(Score);

// Maps input to (.., "-01", "+00", "+01", ..)
std::wstring DoubleDigitSignedInt(Score);

// Maps input to (..,'n', 'u', 'm', '', 'k', 'M', 'G',..)
wchar_t MetricPrefix(int magnitude_base_1000) noexcept(false);

constexpr int64 operator""_kB(uint64 v) noexcept { return v * 1024; }
constexpr int64 operator""_MB(uint64 v) noexcept { return v * 1024 * 1024; }
constexpr int64 operator""_GB(uint64 v) noexcept { return v * 1024 * 1024 * 1024; }
constexpr int64 operator""_TB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024; }
constexpr int64 operator""_EB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024; }
constexpr int64 operator""_ZB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }
constexpr int64 operator""_YB(uint64 v) noexcept { return v * 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 1024; }

std::size_t ParseBytes(const std::string&) noexcept(false);
