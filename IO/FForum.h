#pragma once
#include "Core/Core.h"
#include <vector>

struct PositionScore
{
	Position pos;
	Score score;
};

static std::vector<PositionScore> FForum =
{
	{ Position::Start(), +0 },
	{
		"- - X X X X X -"
		"- O O O X X - O"
		"- O O O X X O X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- X X X O O O -"
		"- O O O O O - -"_pos, +18 // 01
	},
	{
		"- X X X X X X -"
		"- - X O O O O -"
		"- X O X X O O X"
		"X O O O O O O O"
		"O X O O X X O O"
		"O O O X X O O X"
		"- O O O O O - -"
		"- - X X X X X -"_pos, +10 // 02
	},
	{
		"- - - - O X - -"
		"- - O O X X - -"
		"- O O O X X - X"
		"O O X X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X X O X O"
		"- - O O O O O X"_pos, +2 // 03
	},
	{
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X X O X O O O X"
		"- O X O O X X X"
		"- - O O O X X X"
		"- - O O X X - -"
		"- - X O X X O -"_pos, +0 // 04
	},
	{
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O X O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O O - -"
		"- X X X X X - -"_pos, +32 // 05
	},
	{
		"- - O X X X - -"
		"O O O X X X - -"
		"O O O X O X O -"
		"O O X O O O X -"
		"O O X X X X X X"
		"X O O X X O X -"
		"- O O O O X - -"
		"- X X X X X X -"_pos, +14 // 06
	},
	{
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O X X X X"
		"X O O X X X X X"
		"X O O O O X X X"
		"- X X X X X X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 // 07
	},
	{
		"- - - O - O - -"
		"O - O O O O - -"
		"O O O O X O O O"
		"O O O X X X X X"
		"O O X O O O X -"
		"O X O O O O X -"
		"O X X O O O - -"
		"O X X O O X - -"_pos, +8 // 08
	},
	{
		"- - O X O O - -"
		"X - X X O O O O"
		"- X X X O O O O"
		"- O X O O O X O"
		"O O X O X X X O"
		"X O O X O X O O"
		"- - X O X X - -"
		"- - X X X X - -"_pos, -8 // 09
	},
	{
		"- O O O O - - -"
		"- - X O O O - -"
		"O X O X O X O O"
		"X O X O O X O O"
		"X O O X O X X X"
		"O O O X O X X O"
		"- - X O O X - -"
		"- X X X X X - -"_pos, +10 // 10
	},
	{
		"- - - X - O X O"
		"- - - - O O X O"
		"- - - O O X X O"
		"X - O O X O X O"
		"O O O X X O X O"
		"- O X X O O O O"
		"O X X X O O - O"
		"X X X X X X X -"_pos, +30 // 11
	},
	{
		"- - X - - X - -"
		"O - X X X X O -"
		"O O X X X O X X"
		"O O X O X O X X"
		"O O X O O X X X"
		"O O O O X X X X"
		"- - X O O O - -"
		"- O O O O O - -"_pos, -8 // 12
	},
	{
		"- - X X X X X -"
		"- O O O X X - -"
		"- O O O X X X X"
		"- O X O X O X X"
		"O X X X O X X X"
		"- - X O X O X X"
		"- - O X O O O -"
		"- O O O O O - -"_pos, +14 // 13
	},
	{
		"- - X X X X X -"
		"- - O O O X - -"
		"- X O O X X X X"
		"- O O O O O O O"
		"O O O X X X O O"
		"O O O X X O O X"
		"- - X X O O - -"
		"- - X X X X X -"_pos, +18 // 14
	},
	{
		"- - - - O - - -"
		"- - - O O X - -"
		"- O O O X X - X"
		"O O O X O O O O"
		"O X X O X X O O"
		"O X X X O O O O"
		"O X X X O O X O"
		"- - O O O O O X"_pos, +4 // 15
	},
	{
		"- X X X X X X -"
		"X - X X X O O -"
		"X O X X X O O X"
		"X O O X X X O X"
		"- O O O X X X X"
		"- - O O X X X X"
		"- - - O O O - -"
		"- - X O X - O -"_pos, +24 // 16
	},
	{
		"- O O O O O - -"
		"- - O X X O - X"
		"X X O O O X X -"
		"X X O X O X X O"
		"X X O O X O O O"
		"X X X X O O - O"
		"X - X O O - - -"
		"- X X X X - - -"_pos, +8 // 17
	},
	{
		"- X X X - - - -"
		"- - O O O X - -"
		"X O O O O O X X"
		"O X O X O O X X"
		"O X X O O O O O"
		"X X X O X O O X"
		"- - O X X O - -"
		"- O O O O O - -"_pos, -2 // 18
	},
	{
		"- - O X X O - -"
		"X O X X X X - -"
		"X O O O O X X X"
		"X O O O X X X X"
		"X - O O O X X X"
		"- - O O O O X X"
		"- - X X O O O -"
		"- - - X X O O -"_pos, +8 // 19
	},
	{
		"X X X O X X X X"
		"O X X X X X X X"
		"O O X X X X X X"
		"O O O X X X X X"
		"O O O X X O O -"
		"O O O O O - - -"
		"O O O O O O O -"
		"O O O O O O O -"_pos, +6 // 20
	},
	{
		"X X X X X X X X"
		"O X X O O O - -"
		"O O X X O X X -"
		"O X O X X X - -"
		"O X X X X O - -"
		"O X X O X X - -"
		"O X X X X X - -"
		"O O O O - - - -"_pos, +0 // 21
	},
	{
		"- - X X X X - -"
		"O - X X X X X -"
		"O O X X O X O O"
		"O X O X O O O O"
		"O O O X O O O O"
		"- O O X O X O O"
		"- - X O O O - O"
		"- - - - O - - -"_pos, +2 // 22
	},
	{
		"- - O - - - - -"
		"- - O O X - - -"
		"O O O X X X O -"
		"O O O O X O X X"
		"X X X O O X O X"
		"X X X X X O O X"
		"X - X X X X O X"
		"- - X X X X - -"_pos, +4 // 23
	},
	{
		"- - X - - X - -"
		"- - - X X X O -"
		"- O - O X O X X"
		"- - O O O X X X"
		"O O O O X X X X"
		"O O O X O O X X"
		"O O O O O O - -"
		"O X O O - X - -"_pos, +0 // 24
	},
	{
		"- - - - O - - -"
		"- - - O O O X -"
		"- X X X O O O O"
		"O X X X X O O X"
		"- O O X X O O X"
		"X X O X O O O O"
		"X X X O O - - -"
		"O - O O O O - -"_pos, +0 // 25
	},
	{
		"- O O O O O - -"
		"- - O X X O - -"
		"- O O O O X X O"
		"- O O O X O X X"
		"- O O X O O X X"
		"- X O X X O X X"
		"- - O - X X X X"
		"- - O - - - - O"_pos, +0 // 26
	},
	{
		"- - X O - O - -"
		"- - O O O O - -"
		"O O X O X X O -"
		"O O O O X X O O"
		"O O O X X O X -"
		"O X O X X X X X"
		"- - X X X X - -"
		"- - X - O - X -"_pos, -2 // 27
	},
	{
		"- - O - - - - -"
		"- - O O O - - X"
		"- X O O O O X X"
		"X X X X O X O X"
		"- X X O X O O X"
		"X X O X O O X X"
		"- O O O O O - X"
		"- - - O O O - -"_pos, +0 // 28
	},
	{
		"- O X X X X - -"
		"- - O X X O - -"
		"X X O O X O O O"
		"X X X O O X O O"
		"X X O O X O O O"
		"X X X X O O - X"
		"X - X X O - - -"
		"- - - - - - - -"_pos, +10 // 29
	},
	{
		"- X X X - - - -"
		"X - X O O - - -"
		"X X O X O O - -"
		"X O X O X O - -"
		"X O O X O X X X"
		"X O O X X O X -"
		"- - O O O O O -"
		"- X X X X X - -"_pos, +0 // 30
	},
	{
		"- O O O O O - -"
		"- - O O O O - -"
		"O X X O O O - -"
		"- X X X O O - -"
		"X X X X X X O -"
		"X X X O O O - O"
		"X - O O O O - -"
		"- O O O O O - -"_pos, -2 // 31
	},
	{
		"- - X X - - - -"
		"O - X X O X - -"
		"O O X O O - - -"
		"O X O X O O O -"
		"O O X X O O O X"
		"O O X X X O O X"
		"- - X X X X O X"
		"- - X - - X - X"_pos, -4 // 32
	},
	{
		"- X X X X X X X"
		"- - X O O O - -"
		"- - O X O O X X"
		"- O O X X O X X"
		"- O O O O O X X"
		"- X - X O O X X"
		"- - - O - X - X"
		"- - O O O O - -"_pos, -8 // 33
	},
	{
		"- - - - - - - -"
		"- - - - - O - O"
		"- O O O O O O O"
		"O O O O O X O O"
		"O X X O O O X O"
		"- X X X O X O O"
		"- - X X X O X O"
		"- - O X X X X O"_pos, -2 // 34
	},
	{
		"- - O O O - - -"
		"- - O O O O - X"
		"X X O O X X X X"
		"X X X X X X O X"
		"- X X O O O O X"
		"- X X X O O O X"
		"- - - O X O O -"
		"- - O - - - - -"_pos, +0 // 35
	},
	{
		"- - - O - X - -"
		"- - O O O X - O"
		"O O O O O O O O"
		"O X X O O X X O"
		"O X O X X X O O"
		"O O X X X X - O"
		"O - - X X X X -"
		"- - - - - - - -"_pos, +0 // 36
	},
	{
		"- - O O O O - -"
		"O - O O O O - -"
		"O X X X O O O -"
		"O X X O X O - -"
		"O O X X O X X -"
		"O O X X X X - -"
		"O - X X X - - -"
		"- - X X - O - -"_pos, -20 // 37
	},
	{
		"- - O O O O - -"
		"- - O O O O - -"
		"- X O X X O O X"
		"O O X O O O O X"
		"- O O O O O X X"
		"X O O X X X X X"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +4 // 38
	},
	{
		"X - X X X X - -"
		"O X O O X O - -"
		"O X X X O O O -"
		"O X X X O O - -"
		"O X X O X O - -"
		"O X O O O - - -"
		"O - O O - - - -"
		"- - - - - - - -"_pos, +64 // 39
	},
	{
		"O - - O O O O X"
		"- O O O O O O X"
		"O O X X O O O X"
		"O O X O O O X X"
		"O O O O O O X X"
		"- - - O O O O X"
		"- - - - O - - X"
		"- - - - - - - -"_pos, +38 // 40
	},
	{
		"- O O O O O - -"
		"- - O O O O X -"
		"- O O O O O O -"
		"X X X X X O O -"
		"- X X O O X - -"
		"O O X O X X - -"
		"- - O X X O - -"
		"- O O O - - O -"_pos, +0 // 41
	},
	{
		"- - O O O - - -"
		"- - - - X X - O"
		"O O O O O X O O"
		"- O O O O X O O"
		"X - O O O X X O"
		"- - - O O X O O"
		"- - - O O O X O"
		"- - O O O O - -"_pos, +6 // 42
	},
	{
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O O O O -"
		"- X X O O O X -"
		"X X X X O X X -"
		"- - - O X O - -"
		"- - O O O O O -"_pos, -12 // 43
	},
	{
		"- - X - O - X -"
		"- - X - O X - X"
		"- X X O O O X X"
		"X X X X O O O X"
		"X X X X O O - -"
		"O O X X O X - -"
		"- - O O O O - -"
		"- - - O O O - -"_pos, -14 // 44
	},
	{
		"- - - X X X X -"
		"X - X X X O - -"
		"X X O X O O - -"
		"X X X O X O - -"
		"X X O X X O - -"
		"- O X X X O O -"
		"O - O O O O - -"
		"- - - - O O - -"_pos, +6 // 45
	},
	{
		"- - - X X X - -"
		"- - O O O X - -"
		"- - O O O X X -"
		"- O O O O X X X"
		"- - O O O O X X"
		"- - O X O X X X"
		"- - X X O O - -"
		"- X X X X - O -"_pos, -8 // 46
	},
	{
		"- X X X X X - -"
		"- - X X X X - -"
		"- X X X X O - -"
		"O O O O O O - -"
		"- X O X X O - -"
		"X X X O X O - -"
		"- - X X O O - -"
		"- - O O O O - -"_pos, +4 // 47
	},
	{
		"- - - - - O - -"
		"O - O O O - - -"
		"O O O O X X - -"
		"O X O X X O O -"
		"O X X O O O - -"
		"O X X O O - - -"
		"- - X X X O - -"
		"- O O O O O O -"_pos, +28 // 48
	},
	{
		"- - O X - O - -"
		"- - X X O O - -"
		"O O O O O X X -"
		"O O O O O X - -"
		"O O O X O X X -"
		"O O O O X X - -"
		"- - - O O X - -"
		"- - X - O - - -"_pos, +16 // 49
	},
	{
		"- - - - X - - -"
		"- - X X X - - -"
		"- O O O X O O O"
		"- O O O X O O O"
		"- O X O X O X O"
		"- O O X X O O O"
		"- - O O X O - -"
		"- - O - - O - -"_pos, +10 // 50
	},
	{
		"- - - - X - O -"
		"- - - - - O - -"
		"- - - O O O X -"
		"X O O O O O X X"
		"- O O X X O X X"
		"O O X O O O X X"
		"- - X X X X - X"
		"- - - - X X - -"_pos, +6 // 51
	},
	{
		"- - - O - - - -"
		"- - - X O - - O"
		"- - O X X O O O"
		"O O O X O O O O"
		"O O O X X O O O"
		"O O O X X X O O"
		"- - O X - - - O"
		"- - - - - - - -"_pos, +0 // 52
	},
	{
		"- - - - O O - -"
		"- - - O O O - -"
		"- X X X X O O O"
		"- - X X O O X O"
		"- X X X X X O O"
		"- - O O O X O O"
		"- - X - O X - O"
		"- - - - - X - -"_pos, -2 // 53
	},
	{
		"- - O O O - - -"
		"X X O O - - - -"
		"X X X X O O O O"
		"X X X X O X - -"
		"X X X O X X - -"
		"X X O O O - - -"
		"- - - O O O - -"
		"- - - O - - - -"_pos, -2 // 54
	},
	{
		"- - - - - - - -"
		"O - O - - - - -"
		"- O O O O X X X"
		"X X O X O O - -"
		"X X X O O O O -"
		"X X O O O O - -"
		"X - X X X O - -"
		"- - - X X - - -"_pos, +0 // 55
	},
	{
		"- - O O O O O -"
		"- - O O O O - -"
		"- X X X O O - -"
		"- X X O X O - -"
		"- X O O O O O -"
		"X X X X X O X -"
		"- - - X O O - -"
		"- - - - - - - -"_pos, +2 // 56
	},
	{
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - X X O O O"
		"- - X X X O O O"
		"- - X X O X O O"
		"- O O O X X X O"
		"- - O X O O - O"
		"- O O O O O - -"_pos, -10 // 57
	},
	{
		"- - X O O O - -"
		"- - O O O - - -"
		"- O O O X O O -"
		"- O O O O X O -"
		"- O X O X X X -"
		"O O X X X X - -"
		"- - X - X X - -"
		"- - - - - - - -"_pos, +4 // 58
	},
	{
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - O"
		"- - O O O O O -"
		"- - O O O O O X"
		"O O O O X X X X"
		"- - X X O O X X"
		"- - X X - O - X"_pos, +64 // 59
	},
	{
		"- - - O O O O -"
		"- - - O O O - -"
		"- - X O X O X X"
		"- - X O O X X X"
		"- - X O O X X X"
		"- - X O O O X X"
		"- - O X X X - X"
		"- - X X X X - -"_pos, +20 // 60
	},
	{
		"- O O O O - - -"
		"O - O O X O - -"
		"O O O O X O O -"
		"O X X O X X X X"
		"O X X X X X X -"
		"O O X X X X - -"
		"O - - - X - - -"
		"- - - - - - - -"_pos, -14 // 61
	},
	{
		"- - X X X X - -"
		"- - X X O O - -"
		"- - X O O O O O"
		"O O X O O X X X"
		"- O O O O X X -"
		"X O O O O O O X"
		"- - - - O - - -"
		"- - - - - - - -"_pos, +28 // 62
	},
	{
		"- - O - - - - -"
		"- - O - O - - -"
		"- X O O O O - -"
		"- X O O O O X -"
		"X X O X O X X X"
		"X X X X X O X -"
		"- - O X O O - -"
		"- - - O O O O -"_pos, -2 // 63
	},
	{
		"- - X - - O - -"
		"- - X - - O - X"
		"- X X O O O X X"
		"- - O O O O O X"
		"- - O O X X X X"
		"- O O O O O O -"
		"- - O O O - - -"
		"- O - X X X - -"_pos, +20 // 64
	},
	{
		"- - - - O O - -"
		"- - O O O O X -"
		"- - O X X X X -"
		"O - O X X X X -"
		"- O O X X O X -"
		"X X O X X X X -"
		"- - O O O O - -"
		"- - - - - O - -"_pos, +10 // 65
	},
	{
		"- O O O - - - -"
		"X - O X X - - -"
		"X X O X X O O -"
		"X O X X O O - -"
		"X X O O O O - -"
		"X X O O O O - -"
		"- - O O O - - -"
		"- - O - - - - -"_pos, +30 // 66
	},
	{
		"- X X X X X - -"
		"- - X O X X - -"
		"O O O X O X O -"
		"- O O O X O O O"
		"- O O O X X O -"
		"- - O O O X - O"
		"- - - O X - - -"
		"- - - - - - - -"_pos, +22 // 67
	},
	{
		"- - - O O O - -"
		"- - O O O O - -"
		"- - O X X O O X"
		"- O O X X O X -"
		"- O O X X X X -"
		"- X O O X X - -"
		"- - O O O - - -"
		"- - - - - O - -"_pos, +28 // 68
	},
	{
		"- - O O O O - -"
		"- - - O O O - -"
		"- O O O O O - -"
		"X X O X X O O -"
		"- O X O X O O -"
		"O X X X X X X -"
		"- - X - X - - -"
		"- - - - - - - -"_pos, +0 // 69
	},
	{
		"- - - X - - - -"
		"X - X X X - - -"
		"X X X X - - - -"
		"X X X O O O - -"
		"X X X X O O - -"
		"X X O O X X X -"
		"X - O O X X - -"
		"- - O - - - - -"_pos, -24 // 70
	},
	{
		"- - - - - - - -"
		"- - - - - - - -"
		"- - O O O O O -"
		"- O O O O O X -"
		"- X O O O X X O"
		"- - X O X O O O"
		"- - X X O O - O"
		"- - - O O O O -"_pos, +20 // 71
	},
	{
		"- - - X - - - -"
		"- - X X O O - -"
		"- O O X O O O -"
		"O O O O X X O O"
		"- O O O O X X -"
		"- - O O O X X -"
		"- - - O O - - -"
		"- - - - O - - -"_pos, +24 // 72
	},
	{
		"- - O - - O - -"
		"- - O O O - - -"
		"X X O O O O - -"
		"- X X O O O - -"
		"- X O X O O X -"
		"X X X O O O O -"
		"X - - X O X - -"
		"- - - - - - - -"_pos, -4 // 73
	},
	{
		"- - - - O - - -"
		"- - X O O X - O"
		"- - X O X X O O"
		"- X X O O X O O"
		"- - X O O X - O"
		"- - O O X X - -"
		"- - O X X X - -"
		"- - - - - X - -"_pos, -30 // 74
	},
	{
		"- - - - O - - -"
		"- - - - O O - -"
		"- - X X O X - O"
		"- X X X O X O O"
		"- - O O O O O -"
		"- - O O O X O X"
		"- - O O O O X -"
		"- - - - - O - -"_pos, +14 // 75
	},
	{
		"- - - O - - - -"
		"- - O O - O - -"
		"- - - O O O X -"
		"O O O O O O X -"
		"- X X X X O X X"
		"- - O O O O O O"
		"- - O O O - - -"
		"- - - - O - - -"_pos, +32 // 76
	},
	{
		"- - O - O X - -"
		"X - O O O - - -"
		"X X O O O - - -"
		"X X O X O O O O"
		"- O O O O O - -"
		"O - X - O - - -"
		"- - O X - - - -"
		"- - - - - - - -"_pos, +34 // 77
	},
	{
		"- - - - O - - -"
		"- - O O O O - -"
		"- O O O X - X -"
		"O O X O X X X X"
		"- X O O X - - -"
		"X O O O - X - -"
		"- - O O - - - -"
		"- - - O - - - -"_pos, +8 // 78
	},
	{
		"- - - - - - - -"
		"- - - - - - X -"
		"- - - - O - X X"
		"- - - O O O X -"
		"O O O O X O X X"
		"- - O O O O O O"
		"- - O - O O - O"
		"- - - - O O - -"_pos, +64 // 79
	},
};
