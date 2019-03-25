#include "Utility.h"

uint32_t Pow_int(uint32_t base, uint32_t exponent)
{
	if (exponent == 0)
		return 1;
	if (exponent % 2 == 0)
		return Pow_int(base * base, exponent / 2);
	
	return base * Pow_int(base, exponent - 1);
}

//ddd:hh:mm:ss.ccc
std::string time_format(const std::chrono::milliseconds duration)
{
	using days_t = std::chrono::duration<int, std::ratio<24 * 3600> >;
	const auto millis  = duration.count() % 1000;
	const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
	const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
	const auto hours   = std::chrono::duration_cast<std::chrono::hours>  (duration).count() % 24;
	const auto days    = std::chrono::duration_cast<days_t>              (duration).count();

	std::ostringstream oss;
	oss << std::setfill(' ');

	if (days != 0)
		oss << std::setw(3) << days << ":" << std::setfill('0');
	else
		oss << "    ";

	if ((days != 0) || (hours != 0))
		oss << std::setw(2) << hours << ":" << std::setfill('0');
	else
		oss << "   ";

	if ((days != 0) || (hours != 0) || (minutes != 0))
		oss << std::setw(2) << minutes << ":" << std::setfill('0');
	else
		oss << "   ";

	oss << std::setw(2) << seconds << "." << std::setfill('0') << std::setw(3) << millis;

	return oss.str();
}

std::string short_time_format(std::chrono::duration<long long, std::pico> duration)
{
	static const char prefix[] = { 'y', 'z', 'a', 'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M', 'G', 'T', 'P', 'E'};

	const auto ps = duration.count();
	const int magnitude = static_cast<int>(std::floor(std::log10(std::abs(ps)) / 3));
	const double normalized = ps * std::pow(1000.0, -magnitude);

	std::ostringstream oss;
	oss.precision(2 - std::floor(std::log10(std::abs(normalized))));
	oss << std::fixed << std::setw(4) << std::setfill(' ') << normalized << prefix[magnitude + 4] << 's';
	return oss.str();
}

std::string ThousandsSeparator(uint64_t n)
{
	std::ostringstream oss;
	std::locale locale("");
	oss.imbue(locale);
	oss << n;
	return oss.str();
}

std::string DateTimeNow()
{
	std::chrono::system_clock::time_point p = std::chrono::system_clock::now();
	std::time_t t = std::chrono::system_clock::to_time_t(p);
	return std::string(std::ctime(&t));
}

void replace_all(std::string& source, const std::string& from, const std::string& to)
{
	assert(!from.empty());

	for (std::size_t i = source.find(from); i != std::string::npos; i = source.find(from, i + to.length()))
		source.replace(i, from.length(), to);
}

std::vector<std::string> split(const std::string& source, const std::string& delimitter)
{
	assert(!delimitter.empty());
	std::vector<std::string> vec;

	std::size_t begin = 0;
	std::size_t end = source.find(delimitter);
	while (end != std::string::npos) {
		vec.push_back(source.substr(begin, end-begin));
		begin = end + delimitter.length();
		end = source.find(delimitter, begin);
	}
	vec.push_back(source.substr(begin));

	return vec;
}

std::string join(const std::vector<std::string>& parts, const std::string& delimitter)
{
	std::string str;
	for (std::size_t i = 0; i+1 < parts.size(); i++)
		str += parts[i] + delimitter;
	if (!parts.empty())
		str += parts.back();
	return str;
}

std::string SignedInt(int score)
{
	const std::string Sign = (score >= 0) ? "+" : "-";
	const std::string Number = std::to_string(std::abs(score));
	return Sign + Number;
}

std::string DoubleDigitSignedInt(int score)
{
	const std::string Sign = (score >= 0) ? "+" : "-";
	const std::string FillingZero = (std::abs(score) < 10) ? "0" : "";
	const std::string Number = std::to_string(std::abs(score));
	return Sign + FillingZero + Number;
}

std::size_t ParseBytes(const std::string& bytes)
{
	if (bytes.find("EB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("PB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("TB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("GB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024;
	if (bytes.find("MB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024;
	if (bytes.find("kB") != std::string::npos) return std::stoll(bytes) * 1024;
	if (bytes.find( 'B') != std::string::npos) return std::stoll(bytes);
	return 0;
}

uint64_t FlipCodiagonal(uint64_t b)
{
	// 9 x XOR, 6 x SHIFT, 3 x AND
	// 18 OPs

	// # # # # # # # /
	// # # # # # # / #
	// # # # # # / # #
	// # # # # / # # #
	// # # # / # # # #
	// # # / # # # # #
	// # / # # # # # #
	// / # # # # # # #<-LSB
	uint64_t t;
	t  =  b ^ (b << 36);
	b ^= (t ^ (b >> 36)) & 0xF0F0F0F00F0F0F0Fui64;
	t  = (b ^ (b << 18)) & 0xCCCC0000CCCC0000ui64;
	b ^=  t ^ (t >> 18);
	t  = (b ^ (b <<  9)) & 0xAA00AA00AA00AA00ui64;
	b ^=  t ^ (t >>  9);
	return b;
}

uint64_t FlipDiagonal(uint64_t b)
{
	// 9 x XOR, 6 x SHIFT, 3 x AND
	// 18 OPs

	// \ # # # # # # #
	// # \ # # # # # #
	// # # \ # # # # #
	// # # # \ # # # #
	// # # # # \ # # #
	// # # # # # \ # #
	// # # # # # # \ #
	// # # # # # # # \.<-LSB
	uint64_t t;
	t  = (b ^ (b >>  7)) & 0x00AA00AA00AA00AAui64;
	b ^=  t ^ (t <<  7);
	t  = (b ^ (b >> 14)) & 0x0000CCCC0000CCCCui64;
	b ^=  t ^ (t << 14);
	t  = (b ^ (b >> 28)) & 0x00000000F0F0F0F0ui64;
	b ^=  t ^ (t << 28);
	return b;
}

uint64_t FlipHorizontal(uint64_t b)
{
	// 6 x SHIFT, 6 x AND, 3 x OR
	// 15 OPs

	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #
	// # # # #|# # # #<-LSB
	b = ((b >> 1) & 0x5555555555555555ui64) | ((b << 1) & 0xAAAAAAAAAAAAAAAAui64);
	b = ((b >> 2) & 0x3333333333333333ui64) | ((b << 2) & 0xCCCCCCCCCCCCCCCCui64);
	b = ((b >> 4) & 0x0F0F0F0F0F0F0F0Fui64) | ((b << 4) & 0xF0F0F0F0F0F0F0F0ui64);
	return b;
}

uint64_t FlipVertical(uint64_t b)
{
	// 1 x BSwap
	// 1 OPs

	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # #
	// ---------------
	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # #
	// # # # # # # # #<-LSB
	return BSwap(b);
	//b = ((b >>  8) & 0x00FF00FF00FF00FFui64) | ((b <<  8) & 0xFF00FF00FF00FF00ui64);
	//b = ((b >> 16) & 0x0000FFFF0000FFFFui64) | ((b << 16) & 0xFFFF0000FFFF0000ui64);
	//b = ((b >> 32) & 0x00000000FFFFFFFFui64) | ((b << 32) & 0xFFFFFFFF00000000ui64);
	//return b;
}