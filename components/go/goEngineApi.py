import components.go.go as go
import components.go.coords as coords
import numpy as np
import random
import json
import copy
import codecs

BLACK, NONE, WHITE = range(-1, 2)

def selfPlay(board_size, color):
    pos = createGame(board_size, color)
    startGame(pos, color)

def createGame(N, beginner):
    EMPTY_BOARD = np.zeros([5, 5], dtype=np.int8)
    pos = go.Position(EMPTY_BOARD, n=0, komi=7.5, caps=(0, 0),
                      lib_tracker=None, ko=None, recent=tuple(),
                      board_deltas=None, to_play=beginner)
    return pos
# Return pos object, which represents an entire game
def startGame(pos, color):
    while not pos.is_game_over():
        print(pos.board)
        randomNum = random.choice(range(pos.all_legal_moves().size))
        while (pos.all_legal_moves()[randomNum] == 0):
            randomNum = random.choice(range(pos.all_legal_moves().size))
            if not (pos.is_move_legal(coords.from_flat(randomNum)), pos.to_play):
                randomNum = random.choice(range(pos.all_legal_moves().size))
        print(str(color) + " (" + getPlayerName(color) + ") am Zug")
        print("Random Number: " + str(randomNum))
        if (color == WHITE):
            pos = pos.play_move(coords.from_flat(randomNum), WHITE, False)
            color = BLACK
        elif (color == BLACK):
            pos = pos.play_move(coords.from_flat(randomNum), BLACK, False)
            color = WHITE

    print(pos.result())
    print(pos.result_string())
    writeFinalGamestateAsJSON(pos)
    return pos

def getGameState(pos):


    return


def writeFinalGamestateAsJSON(pos):
    position = WriteablePosition(pos.board)
    s = position.getPositionAsJSONString()
    print("test")
    return


def getTrainingSets():
    #TODO auslesen der trainingssets (wo auch immer die liegen)
    return 0

def getPlayerName(color):
    if color == -1:
        return "BLACK"
    elif color == 1:
        return "WHITE"
    else:
        return "FAIL"



class WriteablePosition():
    def __init__(self, board=None):
        self.board = board.tolist() #if board is not None else np.copy(EMPTY_BOARD)
        # With a full history, self.n == len(self.recent) == num moves played

    def getPositionAsJSONString(self):
        return json.dumps(self.__dict__)


