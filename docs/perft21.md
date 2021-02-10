# There are 4'228'388'321'175'157'140 possible Reversi games at the end of the 21-th ply.
[A124004](https://oeis.org/A124004) is the interger sequence of numbers of possible [Reversi](https://en.wikipedia.org/wiki/Reversi) games at the end of the n-th ply.

At the end of the 21-th ply there are **4'228'388'321'175'157'140** possible Reversi games, and here is how you can reproduce this result.

The sequence begins with the starting board

![Image](https://raw.githubusercontent.com/PanicSheep/ReversiPerftCUDA/master/docs/---------------------------OX------XO---------------------------.png)

which has 4 sucessors, one for each red cross. Thus A124004(n=0) = 1 and A124004(n=1) = 4.

The symmetry group of a Reversi board is [Dih4](https://en.wikipedia.org/wiki/Dihedral_group), the same as for a square. There is one additional symmetrie in Reversi. If the colors of the discs are swapped and the color of the player about to play, we arrive at an equivalent position. We use this symmetrie to transform each position so that black is always about to play.

At the end of the 7-th ply there are 55'092 possible Reversi games. Using the transformations from [Dih4](https://en.wikipedia.org/wiki/Dihedral_group) the 55'092 positions can be reduced to a set of 10'649 unique positions. We can now calculate how many games each unique position has at the end of its 14-th ply and associate it with the position. Then we can go through the original 55'092 positions and find their symmetric partner in the unique ones and add the associated numbers up to arrive at A124004(n=21).

The intermediate results listed below can be verified with the code in this project via ```PerftCuda.exe -f perft21_ply7.pos -fd 21 -d 7 -t 1000 1101``` which verifies the positions 1000 to 1100. Additional info is provided with ```-h```. With ```-cuda``` this program makes use of cuda capable accelerators. It took two nVidia GeForce GTX 1080 Ti's with SLI and a Intel Core i9-9900K about 1'000 hours to calculate all positions. So one position took on average 338 s.

All the 10'649 unique positions after the 7-th ply with an arbitrary enumeration are split into these pages:
[0-1'999](https://en.wikipedia.org/wiki/perft21_page1.md)
[2'000-3'999](https://en.wikipedia.org/wiki/perft21_page2.md)
[4'000-5'999](https://en.wikipedia.org/wiki/perft21_page3.md)
[6'000-7'999](https://en.wikipedia.org/wiki/perft21_page4.md)
[8'000-10'648](https://en.wikipedia.org/wiki/perft21_page5.md)
