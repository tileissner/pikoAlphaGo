import components.go.go as go
import components.go.coords as coords
import numpy as np
import random


#from components.go import replayBuffer
from components.go.trainingSet import TrainingSet
from components.mcts.goMCTS import GoGamestate
from components.mcts.search import MonteCarloTreeSearch
from components.mcts.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from utils import constants

BLACK, NONE, WHITE = range(-1, 2)

def selfPlay(board_size, color):
    pos = createGame(board_size, color)
    #trainingSet = startGame(pos, color)
    #return startGame(pos, color)
    return startGameMCTS(pos, color)

def createGame(N, beginner):
    EMPTY_BOARD = np.zeros([5, 5], dtype=np.int8)
    pos = go.Position(EMPTY_BOARD, n=0, komi=0.0, caps=(0, 0),
                      lib_tracker=None, ko=None, recent=tuple(),
                      board_deltas=None, to_play=beginner)
    return pos
# Return pos object, which represents an entire game
def startGame(pos, color):
    trainingSet = []
    while not pos.is_game_over():
        print(pos.board)
        randomNum = random.choice(range(pos.all_legal_moves().size))
        #choseActionAccordingToMCTS(pos)
        while (pos.all_legal_moves()[randomNum] == 0):
            randomNum = random.choice(range(pos.all_legal_moves().size))
            if not (pos.is_move_legal(coords.from_flat(randomNum)), pos.to_play):
                randomNum = random.choice(range(pos.all_legal_moves().size))
        #print(str(color) + " (" + getPlayerName(color) + ") am Zug")v
        currentColor = color
        #print("Random Number: " + str(randomNum))
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
    #print(pos.result())
    #print(pos.result_string())
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



def startGameMCTS(pos, color):

    trainingSet = []
    while not pos.is_game_over():
        print(pos.board)
        actionNumber = choseActionAccordingToMCTS(pos)
        print("gewählte aktion {}", actionNumber)
        #print(str(color) + " (" + getPlayerName(color) + ") am Zug")v
        currentColor = color
        #print("Random Number: " + str(randomNum))
        if (color == WHITE):
            pos = pos.play_move(coords.from_flat(actionNumber), WHITE, False)
            color = BLACK
        elif (color == BLACK):
            pos = pos.play_move(coords.from_flat(actionNumber), BLACK, False)
            color = WHITE
        #TODO: hier ggf. numpy array direkt in normales array umwandeln
        newTrainingSet = TrainingSet(pos.board, getMockProbabilities(pos), currentColor)
        trainingSet.append(newTrainingSet)

    #update winner when game is finished for all experiences in this single game
    for newTrainingSet in trainingSet:
        newTrainingSet.updateWinner(pos.result())

    winner = pos.result()
    #print(pos.result())
    #print(pos.result_string())
    #replayBuffer.addToReplayBuffer(trainingSet)
    #return pos
    return trainingSet

def choseActionAccordingToMCTS(pos):
    state = pos.board
    initial_board_state = GoGamestate(pos.board, constants.board_size, pos.to_play, pos)

    root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state,
                                                  parent=None)

    mcts = MonteCarloTreeSearch(root)
    resultChild = mcts.best_action(1000)
    return getActionFromNode(resultChild, pos)

def getActionFromNode(node, pos):

    # print(pos.board)
    # print("vs")
    # print(node.state.board)

    # for x in node.state.board.flatten():
    #     print(x)
    #TODO: Not sure ob die Funktion 100% einwandfrei läuft (wegen array indezes und so)
    differences = []
    for index, values in np.ndenumerate(node.state.board.flatten()):
        if (pos.board.flatten()[index[0]] != values):
            print("found " + str(index[0]))
            differences.append(index[0])

    #differences = np.where(pos.board!=node.state.board)
    print(differences)
    if len(differences) > 1:
        raise ValueError("Differences after MCTS must not be more than one stone")
    print(coords.from_flat(differences[0]))
    #return coords.from_flat(differences[0])
    return differences[0]