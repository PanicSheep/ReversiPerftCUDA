#include "IO.h"
#include "Integers.h"
#include "Core/Core.h"
#include <array>
#include <sstream>

static const std::array<std::wstring, 65> field_names = {
	L"A1", L"B1", L"C1", L"D1", L"E1", L"F1", L"G1", L"H1",
	L"A2", L"B2", L"C2", L"D2", L"E2", L"F2", L"G2", L"H2",
	L"A3", L"B3", L"C3", L"D3", L"E3", L"F3", L"G3", L"H3",
	L"A4", L"B4", L"C4", L"D4", L"E4", L"F4", L"G4", L"H4",
	L"A5", L"B5", L"C5", L"D5", L"E5", L"F5", L"G5", L"H5",
	L"A6", L"B6", L"C6", L"D6", L"E6", L"F6", L"G6", L"H6",
	L"A7", L"B7", L"C7", L"D7", L"E7", L"F7", L"G7", L"H7",
	L"A8", L"B8", L"C8", L"D8", L"E8", L"F8", L"G8", L"H8", L"--"
};

std::wstring to_wstring(Field f) noexcept
{
	return field_names[static_cast<uint8_t>(f)];
}

Field ParseField(const std::wstring& str) noexcept
{
	const auto it = std::find(field_names.begin(), field_names.end(), str);
	const auto index = std::distance(field_names.begin(), it);
	return static_cast<Field>(index);
}

std::wstring SingleLine(const Position& pos)
{
	std::wstring str = L"---------------------------------------------------------------- X";
	for (int i = 0; i < 64; i++)
	{
		if (pos.Player().Get(static_cast<Field>(63 - i)))
			str[i] = L'X';
		else if (pos.Opponent().Get(static_cast<Field>(63 - i)))
			str[i] = L'O';
	}
	return str;
}

std::wstring SingleLine(const BitBoard& bb)
{
	std::wstring str = L"----------------------------------------------------------------";
	for (int i = 0; i < 64; i++)
		if (bb.Get(static_cast<Field>(63 - i)))
			str[i] = L'#';
	return str;
}

std::wstring MultiLine(const Position& pos)
{
	Moves moves = PossibleMoves(pos);
	std::wstring puzzle =
		L"  H G F E D C B A  \n"
		L"8 - - - - - - - - 8\n"
		L"7 - - - - - - - - 7\n"
		L"6 - - - - - - - - 6\n"
		L"5 - - - - - - - - 5\n"
		L"4 - - - - - - - - 4\n"
		L"3 - - - - - - - - 3\n"
		L"2 - - - - - - - - 2\n"
		L"1 - - - - - - - - 1\n"
		L"  H G F E D C B A  ";

	for (int i = 0; i < 64; i++)
	{
		auto& field = puzzle[22 + 2 * i + 4 * (i / 8)];

		if (pos.Player().Get(static_cast<Field>(63 - i)))
			field = 'X';
		else if (pos.Opponent().Get(static_cast<Field>(63 - i)))
			field = 'O';
		else if (moves.contains(static_cast<Field>(63 - i)))
			field = '+';
	}
	return puzzle;
}

std::wstring MultiLine(const BitBoard& bb)
{
	std::wstring puzzle =
		L"  H G F E D C B A  \n"
		L"8 - - - - - - - - 8\n"
		L"7 - - - - - - - - 7\n"
		L"6 - - - - - - - - 6\n"
		L"5 - - - - - - - - 5\n"
		L"4 - - - - - - - - 4\n"
		L"3 - - - - - - - - 3\n"
		L"2 - - - - - - - - 2\n"
		L"1 - - - - - - - - 1\n"
		L"  H G F E D C B A  ";

	for (int i = 0; i < 64; i++)
		if (bb.Get(static_cast<Field>(63 - i)))
			puzzle[22 + 2 * i + 4 * (i / 8)] = '#';
	return puzzle;
}

Position ParsePosition_SingleLine(const std::wstring& str) noexcept(false)
{
	BitBoard P, O;

	for (int i = 0; i < 64; i++)
	{
		if (str[i] == L'X')
			P.Set(static_cast<Field>(63 - i));
		if (str[i] == L'O')
			O.Set(static_cast<Field>(63 - i));
	}

	if (str[65] == L'X')
		return Position{ P, O };
	if (str[65] == L'O')
		return Position{ O, P };
	throw;
}

// Format: "ddd:hh:mm:ss.ccc"
std::wstring time_format(const std::chrono::milliseconds duration)
{
	using namespace std;
	using days_t = chrono::duration<int, ratio<24 * 3600>>;

	const auto milli_seconds = duration.count() % 1000;
	const auto seconds = chrono::duration_cast<chrono::seconds>(duration).count() % 60;
	const auto minutes = chrono::duration_cast<chrono::minutes>(duration).count() % 60;
	const auto hours = chrono::duration_cast<chrono::hours>(duration).count() % 24;
	const auto days = chrono::duration_cast<days_t>(duration).count();

	wostringstream oss;
	oss << setfill(L' ');

	if (days)
		oss << setw(3) << days << L':'<< setfill(L'0');
	else
		oss << L"    ";

	if (days || hours)
		oss << setw(2) << hours << L':'<< setfill(L'0');
	else
		oss << L"   ";

	if (days || hours || minutes)
		oss << setw(2) << minutes << L':'<< setfill(L'0');
	else
		oss << L"   ";

	oss << setw(2) << seconds << L'.' << setfill(L'0') << setw(3) << milli_seconds;

	return oss.str();
}

std::wstring short_time_format(std::chrono::duration<double> duration)
{
	using namespace std;

	const auto seconds = duration.count();
	const int magnitude_base_1000 = static_cast<int>(floor(log10(abs(seconds)) / 3));
	const double normalized = seconds * pow(1000.0, -magnitude_base_1000);

	wostringstream oss;
	oss.precision(2 - floor(log10(abs(normalized))));
	oss << fixed << setw(4) << setfill(L' ') << normalized << MetricPrefix(magnitude_base_1000) << L's';
	return oss.str();
}
