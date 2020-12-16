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


print(pos.is_game_over())
print(pos.board)
print("GAME OVER!")
