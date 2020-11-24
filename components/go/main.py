import components.go.go as go
import numpy as np
import components.go.coords as coords
import random

BLACK, NONE, WHITE = range(-1, 2)
N=5
EMPTY_BOARD = np.zeros([5, 5], dtype=np.int8)
pos = go.Position(EMPTY_BOARD, n=0, komi=7.5, caps=(0, 0),
                      lib_tracker=None, ko=None, recent=tuple(),
                      board_deltas=None, to_play=BLACK)

color=BLACK
while not pos.is_game_over():
    print(pos.board)
    randomNum = random.choice(range(pos.all_legal_moves().size))
    while(pos.all_legal_moves()[randomNum]==0):
        randomNum = random.choice(range(pos.all_legal_moves().size))
        if not (pos.is_move_legal(coords.from_flat(randomNum)), pos.to_play):
            randomNum = random.choice(range(pos.all_legal_moves().size))
    print(str(color) + " to move")
    print("Random Number: " + str(randomNum))
    if(color == WHITE):
        pos=pos.play_move(coords.from_flat(randomNum), WHITE, False)
        color = BLACK
    elif(color == BLACK):
        pos = pos.play_move(coords.from_flat(randomNum), BLACK, False)
        color = WHITE



# pos=pos.play_move(coords.from_flat(0))
# pos=pos.play_move(coords.from_flat(4))
# pos=pos.play_move(coords.from_flat(2))
# pos=pos.play_move(coords.from_flat(3))
# pos=pos.play_move(coords.from_flat(1))
# pos=pos.play_move(coords.from_flat(6))
# pos=pos.play_move(coords.from_flat(8))
# pos=pos.play_move(coords.from_flat(10))
# pos=pos.play_move(coords.from_flat(7))
# pos=pos.play_move(coords.from_flat(9))
# pos=pos.play_move(coords.from_flat(5))
# pos=pos.play_move(coords.from_flat(12))
# pos=pos.play_move(coords.from_flat(16))
# pos=pos.play_move(coords.from_flat(18))
# pos=pos.play_move(coords.from_flat(13))
# pos=pos.play_move(coords.from_flat(23))
# pos=pos.play_move(coords.from_flat(15))
# print(pos.board)
# pos=pos.play_move(coords.from_flat(14))

print(pos.is_game_over())
print(pos.board)
print("GAME OVER!")
