import struct
from PIL import Image, ImageDraw
import locale
locale.setlocale(locale.LC_ALL, '')

class BitBoard:
    def __init__(self, b=0):
        self.__b = b

    def __eq__(self, o):
        return self.__b == o.__b
    def __neq__(self, o):
        return self.__b != o.__b

    def __and__(self, o):
        return BitBoard(self.__b & o.__b)
    def __or__(self, o):
        return BitBoard(self.__b | o.__b)
    def __invert__(self):
        return BitBoard(~self.__b)

    def __lshift__(self, i):
        return BitBoard(self.__b << i)
    def __rshift__(self, i):
        return BitBoard(self.__b >> i)

    def __getitem__(self, key) -> bool:
        x,y = key
        return bool(self.__b & (1 << (63 - x - 8 * y)))
    def __setitem__(self, key, value:bool):
        x,y = key
        if value:
            self.__b |= (1 << (63 - x - 8 * y))
        else:
            self.__b &= ~(1 << (63 - x - 8 * y))

class Position:
    def __init__(self, P=BitBoard(), O=BitBoard()):
        if not isinstance(P, BitBoard):
            P = BitBoard(P)
        if not isinstance(O, BitBoard):
            O = BitBoard(O)
        self.P = P
        self.O = O

def PlayPass(pos:Position):
    return Position(pos.O, pos.P)

def FlipsInOneDirection(pos:Position, x, y, dx, dy) -> BitBoard:
    flips = BitBoard()
    x += dx
    y += dy
    while (x >= 0) and (x < 8) and (y >= 0) and (y < 8):
        if pos.O[x,y]:
            flips[x,y] = True
        elif pos.P[x,y]:
            return flips
        else:
            break
        x += dx
        y += dy
    return BitBoard()

def Flips(pos:Position, x, y) -> BitBoard:
    return FlipsInOneDirection(pos, x, y, -1, -1) \
         | FlipsInOneDirection(pos, x, y, -1, +0) \
         | FlipsInOneDirection(pos, x, y, -1, +1) \
         | FlipsInOneDirection(pos, x, y, +0, -1) \
         | FlipsInOneDirection(pos, x, y, +0, +1) \
         | FlipsInOneDirection(pos, x, y, +1, -1) \
         | FlipsInOneDirection(pos, x, y, +1, +0) \
         | FlipsInOneDirection(pos, x, y, +1, +1)

def PossibleMoves(pos:Position) -> BitBoard:
    maskO = pos.O & BitBoard(0x7E7E7E7E7E7E7E7E)
    
    flip1 = maskO & (pos.P << 1);
    flip2 = maskO & (pos.P >> 1);
    flip3 = pos.O & (pos.P << 8);
    flip4 = pos.O & (pos.P >> 8);
    flip5 = maskO & (pos.P << 7);
    flip6 = maskO & (pos.P >> 7);
    flip7 = maskO & (pos.P << 9);
    flip8 = maskO & (pos.P >> 9);

    flip1 |= maskO & (flip1 << 1);
    flip2 |= maskO & (flip2 >> 1);
    flip3 |= pos.O & (flip3 << 8);
    flip4 |= pos.O & (flip4 >> 8);
    flip5 |= maskO & (flip5 << 7);
    flip6 |= maskO & (flip6 >> 7);
    flip7 |= maskO & (flip7 << 9);
    flip8 |= maskO & (flip8 >> 9);

    mask1 = maskO & (maskO << 1);
    mask2 = mask1 >> 1;
    mask3 = pos.O & (pos.O << 8);
    mask4 = mask3 >> 8;
    mask5 = maskO & (maskO << 7);
    mask6 = mask5 >> 7;
    mask7 = maskO & (maskO << 9);
    mask8 = mask7 >> 9;

    flip1 |= mask1 & (flip1 << 2);
    flip2 |= mask2 & (flip2 >> 2);
    flip3 |= mask3 & (flip3 << 16);
    flip4 |= mask4 & (flip4 >> 16);
    flip5 |= mask5 & (flip5 << 14);
    flip6 |= mask6 & (flip6 >> 14);
    flip7 |= mask7 & (flip7 << 18);
    flip8 |= mask8 & (flip8 >> 18);

    flip1 |= mask1 & (flip1 << 2);
    flip2 |= mask2 & (flip2 >> 2);
    flip3 |= mask3 & (flip3 << 16);
    flip4 |= mask4 & (flip4 >> 16);
    flip5 |= mask5 & (flip5 << 14);
    flip6 |= mask6 & (flip6 >> 14);
    flip7 |= mask7 & (flip7 << 18);
    flip8 |= mask8 & (flip8 >> 18);

    flip1 <<= 1;
    flip2 >>= 1;
    flip3 <<= 8;
    flip4 >>= 8;
    flip5 <<= 7;
    flip6 >>= 7;
    flip7 <<= 9;
    flip8 >>= 9;

    return ~(pos.P | pos.O) & (flip1 | flip2 | flip3 | flip4 | flip5 | flip6 | flip7 | flip8);

def to_string(pos:Position):
    ret = ''
    for y in range(8):
        for x in range(8):
            if pos.P[x,y]:
                ret += 'X'
            elif pos.O[x,y]:
                ret += 'O'
            else:
                ret += '-'
    return ret

def to_png(pos:Position):
    d = 10
    out = Image.new("RGB", (8*d+1,8*d+1), (0,0,0))
    draw = ImageDraw.Draw(out)

    # Board
    for x in range(8):
        for y in range(8):
            draw.rectangle([x*d, y*d, (x+1)*d, (y+1)*d], (0,100,0), (0,50,0))
    # Small black dots
    for x in [2,6]:
        for y in [2,6]:
            draw.ellipse([x*d-0.05*d, y*d-0.05*d, x*d+0.05*d+1, y*d+0.05*d+1], (0,50,0), (0,50,0))

    possible_moves = PossibleMoves(pos)
    for x in range(8):
        for y in range(8):
            if pos.P[x,y]:
                draw.ellipse([x*d+0.075*d+1, y*d+0.075*d+1, (x+1)*d-0.075*d, (y+1)*d-0.075*d], (0,0,0), (0,0,0))
            elif pos.O[x,y]:
                draw.ellipse([x*d+0.075*d+1, y*d+0.075*d+1, (x+1)*d-0.075*d, (y+1)*d-0.075*d], (255,255,255), (255,255,255))
            elif possible_moves[x,y]:
                draw.line([x*d+0.4*d, y*d+0.4*d, (x+1)*d-0.4*d, (y+1)*d-0.4*d], (255,0,0), int(d/20))
                draw.line([x*d+0.4*d, (y+1)*d-0.4*d, (x+1)*d-0.4*d, y*d+0.4*d], (255,0,0), int(d/20))

    out.save(f'G:\\Reversi\\perft\\ply7\\{to_string(pos)}.png')

def DataGenerator():
    chunk_size = struct.calcsize('<QQQQ')
    with open(f'G:\\Reversi\\perft\\perft21_ply7.pos', "rb") as file:
        while True:
            try:
                P, O, value, _ = struct.unpack('<QQQQ', file.read(chunk_size))
                yield Position(P, O), value
            except:
                break

if __name__ == '__main__':
    with open('perft21_ply7.txt', 'w') as f:
        for i, (pos, value) in enumerate(DataGenerator()):
            #to_png(pos)
            print(f'|{i}|![Image](https://raw.githubusercontent.com/PanicSheep/ReversiPerftCUDA/master/docs/ply7/{to_string(pos)}.png)|{value:n}|', file=f)



