import components.go.go as go
import components.go.coords as coords
import numpy as np
import random


#from components.go import replayBuffer
from components.go.trainingSet import TrainingSet

BLACK, NONE, WHITE = range(-1, 2)

def selfPlay(board_size, color):
    pos = createGame(board_size, color)
    #trainingSet = startGame(pos, color)
    return startGame(pos, color)

def createGame(N, beginner):
    EMPTY_BOARD = np.zeros([5, 5], dtype=np.int8)
    pos = go.Position(EMPTY_BOARD, n=0, komi=7.5, caps=(0, 0),
                      lib_tracker=None, ko=None, recent=tuple(),
                      board_deltas=None, to_play=beginner)
    return pos
# Return pos object, which represents an entire game
def startGame(pos, color):
    trainingSet = []
    while not pos.is_game_over():
        print(pos.board)
        randomNum = random.choice(range(pos.all_legal_moves().size))
        while (pos.all_legal_moves()[randomNum] == 0):
            randomNum = random.choice(range(pos.all_legal_moves().size))
            if not (pos.is_move_legal(coords.from_flat(randomNum)), pos.to_play):
                randomNum = random.choice(range(pos.all_legal_moves().size))
        print(str(color) + " (" + getPlayerName(color) + ") am Zug")
        currentColor = color
        print("Random Number: " + str(randomNum))
        if (color == WHITE):
            pos = pos.play_move(coords.from_flat(randomNum), WHITE, False)
            color = BLACK
        elif (color == BLACK):
            pos = pos.play_move(coords.from_flat(randomNum), BLACK, False)
            color = WHITE
        #TODO: hier ggf. numpy array direkt in normales array umwandeln
        newTrainingSet = TrainingSet(pos.board, getMockProbabilities(pos), currentColor)
        trainingSet.append(newTrainingSet)

    #update winner when game is finished for all experiences in this single game
    for newTrainingSet in trainingSet:
        newTrainingSet.updateWinner(pos.result())

    winner = pos.result()
    print(pos.result())
    print(pos.result_string())
    #replayBuffer.addToReplayBuffer(trainingSet)
    #return pos
    return trainingSet

# def createJsonObject(trainingSet):
#     jsonObject = {}
#     for t in trainingSet:
#         jsonObject.update




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

def getMockProbabilities(pos):
    probabilities = {}
    #test = pos.all_legal_moves()
    #print(pos.all_legal_moves())

    #Performance Update: Without if statement?

    for i in range(0, pos.all_legal_moves().size):
        if i != 0:
            probabilities.update({i: random.random()})
        else:
            probabilities.update({i: 0.0})


    return probabilities




