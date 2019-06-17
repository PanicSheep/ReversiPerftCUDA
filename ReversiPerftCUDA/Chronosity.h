#pragma once
#include <memory>
#include <type_traits>

enum chronosity { syn, asyn };

template<class _Ty>
__host__ __device__ inline void swap(_Ty& _Left, _Ty& _Right) noexcept(std::is_nothrow_move_constructible_v<_Ty> && std::is_nothrow_move_assignable_v<_Ty>)
{
	_Ty _Tmp = std::move(_Left);
	_Left = std::move(_Right);
	_Right = std::move(_Tmp);
}