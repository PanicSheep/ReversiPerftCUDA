#include "Player.h"

Position RandomPlayer::Play(const Position& pos)
{
	Moves possible_moves = PossibleMoves(pos);
	if (!possible_moves)
		return PlayPass(pos);

	auto rnd = std::uniform_int_distribution<std::size_t>(0, possible_moves.size())(rnd_engine);
	for (std::size_t i = 0; i < rnd; i++)
		possible_moves.RemoveFirst();

	return ::Play(pos, possible_moves.First());
}
