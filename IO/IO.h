#pragma once
#include "Core/Core.h"
#include "FForum.h"
#include "Integers.h"
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <memory>

// Maps 'Field::A1' -> "A1", ... , 'Field::invalid' -> "--".
std::wstring to_wstring(Field) noexcept;
Field ParseField(const std::wstring&) noexcept;

std::wstring SingleLine(const Position&);
std::wstring SingleLine(const BitBoard&);
std::wstring MultiLine(const Position&);
std::wstring MultiLine(const BitBoard&);

Position ParsePosition_SingleLine(const std::wstring&) noexcept(false);



// Format: "ddd:hh:mm:ss.ccc"
std::wstring time_format(const std::chrono::milliseconds duration);

// Format: "ddd:hh:mm:ss.ccc"
template <class U, class V>
std::wstring time_format(std::chrono::duration<U, V> duration)
{
	return time_format(std::chrono::duration_cast<std::chrono::milliseconds>(duration));
}

std::wstring short_time_format(std::chrono::duration<double> duration);

template <typename Iterator>
void WriteToFile(const std::filesystem::path& file, const Iterator& begin, const Iterator& end)
{
	std::fstream stream(file, std::ios::out | std::ios::binary);
	if (!stream.is_open())
		throw std::fstream::failure("Can not open '" + file.string() + "' for binary output.");

	for (auto it = begin; it != end; ++it)
		stream.write(reinterpret_cast<const char*>(std::addressof(*it)), sizeof(std::iterator_traits<Iterator>::value_type));
}

template <typename Container>
void WriteToFile(const std::filesystem::path& file, const Container& c)
{
	WriteToFile(file, c.cbegin(), c.cend());
}

class PosScoreFileStream
{
	#pragma pack(1)
	struct DensePosScore
	{
		Position pos{};
		int8_t score{};
	};
	#pragma pack()

	std::fstream stream;
public:
	PosScoreFileStream(const std::filesystem::path& file) : stream(file, std::ios::in | std::ios::binary)
	{
		if (!stream.is_open())
			throw std::fstream::failure("Can not open '" + file.string() + "' for binary intput.");
	}

	[[nodiscard]] bool eof() noexcept { stream.peek(); return stream.eof(); }

	[[nodiscard]] PositionScore read()
	{
		DensePosScore buffer;
		stream.read(reinterpret_cast<char*>(&buffer), sizeof(DensePosScore));
		return { buffer.pos, buffer.score };
	}
};

template <typename OutputIt>
void ReadPosScoreFile(OutputIt first, const std::filesystem::path& file, std::size_t count = std::numeric_limits<std::size_t>::max())
{
	PosScoreFileStream stream(file);

	while (count-- > 0 && !stream.eof())
		*first++ = stream.read();
}