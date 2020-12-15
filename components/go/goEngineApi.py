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
from components.mcts.stanfordmcts import search

BLACK, NONE, WHITE = range(-1, 2)

def selfPlay(board_size, color):
    pos = createGame(board_size, color)
    #trainingSet = startGame(pos, color)
    #return startGame(pos, color)
    return startGameMCTS(pos, color)
    #return startGameStanfordMCTS(pos)

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
    #probabilities = {}
    probabilities = []

    #Performance Update: Without if statement?

    #for i in range(0, pos.all_legal_moves().size):
    index = 0
    for move in pos.all_legal_moves():
        if move == 1:
            #probabilities.update({index: random.random()})
            if index == (constants.board_size * constants.board_size):
                #TODO fixen, sobald richtige werte da sind
                probabilities.append(0.02)
            else:
                probabilities.append(random.random())
        else:
            #probabilities.update({index: 0.0})
            probabilities.append(0.0)
        index += 1


    return probabilities



def startGameMCTS(pos, color):

    trainingSet = []
    while not pos.is_game_over():
        print(pos.board)
        action = choseActionAccordingToMCTS(pos)
        print("gewählte aktion ", action)
        #print(str(color) + " (" + getPlayerName(color) + ") am Zug")v
        currentColor = color
        #print("Random Number: " + str(randomNum))
        if (color == WHITE):
            pos = pos.play_move(action, WHITE, False)
            color = BLACK
        elif (color == BLACK):
            pos = pos.play_move(action, BLACK, False)
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

def startGameStanfordMCTS(pos):

    trainingSet = []
    while not pos.is_game_over():
        for i in range(constants.mcts_simulations):
            search(pos.board, pos, None)


        newTraining = TrainingSet(pos.board, getMockProbabilities(pos), pos.to_play)
        trainingSet.append(newTraining)

        a = newTraining.getBestActionFromProbabilities()
        print("gewählte aktion {}", a)
        pos = pos.play_move(coords.from_flat(a))

        print(pos.board)


    #update winner when game is finished for all experiences in this single game
    for t in trainingSet:
        t.updateWinner(pos.result())

    winner = pos.result()
    print("winner: " + str(winner))
    return trainingSet

# def choseActionAccordingToMCTS(pos):
#     state = pos.board
#     search(state, pos, None)
#     return getActionFromNode(resultChild, pos)

def choseActionAccordingToMCTS(pos):
    initial_board_state = GoGamestate(pos.board, constants.board_size, pos.to_play, pos)

    root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state, move_from_parent=None,
                                                  parent=None)

    mcts = MonteCarloTreeSearch(root)
    #resultChild = mcts.best_action(1000)
    resultChild = mcts.search_function(constants.mcts_simulations)
    return coords.from_flat(resultChild.move_from_parent)
    #return getActionFromNode(resultChild, pos)

# def getActionFromNode(node, pos):
# ''''funktion vermutlich trash, weil sich durch einen move viele felder verändern können'''''
#     # print(pos.board)
#     # print("vs")
#     # print(node.state.board)
#
#     # for x in node.state.board.flatten():
#     #     print(x)
#     #TODO: Not sure ob die Funktion 100% einwandfrei läuft (wegen array indezes und so)
#     differences = []
#     for index, values in np.ndenumerate(node.state.board.flatten()):
#         if (pos.board.flatten()[index[0]] != values):
#             print("found " + str(index[0]))
#             differences.append(index[0])
#
#     #differences = np.where(pos.board!=node.state.board)
#     #print(differences)
#     action = None
#
#     if len(differences) > 1:
#         print("test")
#         raise ValueError("Differences after MCTS must not be more than one stone")
#     if len(differences) == 0:
#         action = coords.from_flat(constants.board_size * constants.board_size) #pass wenn sich die spielstände nicht verändert haben
#         print("Achtung: Pass wurde ausgewählt, da die Spielstände sich nicht unterscheiden")
#     else:
#         action = coords.from_flat(differences[0])
#     #return coords.from_flat(differences[0])
#     return action