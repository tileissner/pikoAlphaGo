import random

import numpy as np

import components.go.coords as coords
import components.go.go as go
from components.go.trainingSet import TrainingSet
from components.mcts.goMCTS import GoGamestate
from components.mcts.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from components.mcts.search import MonteCarloTreeSearch
from components.nn.nn_api import NetworkAPI
from components.player.player import Player
from utils import constants

BLACK, NONE, WHITE = range(-1, 2)


def evaluateNet(board_size, color, currentNetFileName, challengerNetFileName):
    pos = createGame(board_size, color)
    currentPlayerWins = 0
    challengerPlayerWins = 0

    currentPlayer = Player(WHITE, currentNetFileName)
    challengerPlayer = Player(BLACK, challengerNetFileName)

    for _ in range(0, constants.amount_evaluator_iterations):
        # Zufällige Auswahl wer beginnt
        color = randomStartPlayer()

        winner = startGameEvaluation(pos, color, currentPlayer, challengerPlayer)
        if winner == currentPlayer:
            currentPlayerWins += 1
        else:
            challengerPlayerWins += 1

    print(str(currentPlayer.color) + " wins")
    print(str(challengerPlayer.color) + " wins")

    # wenn 55% -> neues model
    if challengerPlayerWins / constants.amount_evaluator_iterations > 0.55:
        print("neues netz ist besser!")
        # überschreibe das current best network mit dem neuen challenger network
        constants.currentBestNetFileName = challengerPlayer.net_api.pathToModel
    else:
        print("neues netz bringt keine verbesserung")



def selfPlay(board_size, color):
    pos = createGame(board_size, color)
    # trainingSet = startGame(pos, color)
    # return startGame(pos, color)
    return startGameMCTS(pos, color)


def createGame(N, beginner):
    EMPTY_BOARD = np.zeros([N, N], dtype=np.int8)
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
        # choseActionAccordingToMCTS(pos)
        while (pos.all_legal_moves()[randomNum] == 0):
            randomNum = random.choice(range(pos.all_legal_moves().size))
            if not (pos.is_move_legal(coords.from_flat(randomNum)), pos.to_play):
                randomNum = random.choice(range(pos.all_legal_moves().size))
        # print(str(color) + " (" + getPlayerName(color) + ") am Zug")v
        currentColor = color
        # print("Random Number: " + str(randomNum))
        if (color == WHITE):
            pos = pos.play_move(coords.from_flat(randomNum), WHITE, False)
            color = BLACK
        elif (color == BLACK):
            pos = pos.play_move(coords.from_flat(randomNum), BLACK, False)
            color = WHITE
        newTrainingSet = TrainingSet(pos.board, getMockProbabilities(pos), currentColor)
        trainingSet.append(newTrainingSet)

    # update winner when game is finished for all experiences in this single game
    for newTrainingSet in trainingSet:
        newTrainingSet.updateWinner(pos.result())

    winner = pos.result()
    # print(pos.result())
    # print(pos.result_string())
    # replayBuffer.addToReplayBuffer(trainingSet)
    # return pos
    return trainingSet


def getTrainingSets():
    # TODO auslesen der trainingssets (wo auch immer die liegen)
    return 0


def getPlayerName(color):
    if color == -1:
        return "BLACK"
    elif color == 1:
        return "WHITE"
    else:
        return "FAIL"


def getMockRealProbabilities(pos):
    # probabilities = []

    # Performance Update: Without if statement?

    # for i in range(0, pos.all_legal_moves().size):
    validIndeces = []

    index = 0
    for move in pos.all_legal_moves():
        if move == 1:
            validIndeces.append(index)
        index += 1
    allMoveArray = pos.all_legal_moves()

    allMoveArray = [float(item) for item in allMoveArray]
    probabilities = np.random.dirichlet(np.ones(len(validIndeces)), size=1).flatten()
    index = 0
    probabilities = probabilities.tolist()
    for i in range(len(allMoveArray)):
        if allMoveArray[i] != 0:
            allMoveArray[i] = probabilities[index]
            index += 1

    return allMoveArray


def getMockProbabilities(pos):
    # probabilities = {}
    probabilities = []

    # Performance Update: Without if statement?

    # for i in range(0, pos.all_legal_moves().size):
    index = 0
    for move in pos.all_legal_moves():
        if move == 1:
            # probabilities.update({index: random.random()})
            if index == (constants.board_size * constants.board_size):
                # TODO fixen, sobald richtige werte da sind
                probabilities.append(0.00001)
            else:
                probabilities.append(random.random())
        else:
            # probabilities.update({index: 0.0})
            probabilities.append(0.0)
        index += 1

    return probabilities


def zero_illegal_moves_from_prediction(pos, probs):
    probabilities = []

    index = 0
    for move in pos.all_legal_moves():
        if move == 0:
            probabilities.append(0.0)
        else:  # legal move
            # probabilities.update({index: 0.0})
            probabilities.append(abs(probs[index]))
        index += 1

    return probabilities


def startGameMCTS(pos, color):
    trainingSet = []

    # Temoporärer Ladevorgang d Netzes

    net_api = NetworkAPI()
    net_api.model_load(constants.currentBestNetFileName)

    while not pos.is_game_over():
        # print(pos.board)
        action, probs = chooseActionAccordingToMCTS(pos, net_api)

        #GetMoveProbablitiesFrom MCTS


        print("gewählte aktion ", action)
        # print(str(color) + " (" + getPlayerName(color) + ") am Zug")v
        currentColor = color
        # print("Random Number: " + str(randomNum))
        if (color == WHITE):
            pos = pos.play_move(action, WHITE, False)
            color = BLACK
        elif (color == BLACK):
            pos = pos.play_move(action, BLACK, False)
            color = WHITE
        # TODO: hier ggf. numpy array direkt in normales array umwandeln
        mockprobs = getMockProbabilities(pos)

        newTrainingSet = TrainingSet(pos.board, probs, currentColor)
        trainingSet.append(newTrainingSet)

    # update winner when game is finished for all experiences in this single game
    for newTrainingSet in trainingSet:
        newTrainingSet.updateWinner(pos.result())

    winner = pos.result()
    # print(pos.result())
    # print(pos.result_string())
    # replayBuffer.addToReplayBuffer(trainingSet)
    # return pos
    return trainingSet


def startGameEvaluation(pos, color, currentPlayer, challengerPlayer):
    while not pos.is_game_over():
        # print(str(color) + " (" + getPlayerName(color) + ") am Zug")v

        # print("Random Number: " + str(randomNum))
        if (color == WHITE):
            action, probs = chooseActionAccordingToMCTS(pos, currentPlayer.net_api)
            #print("gewählte aktion ", action)
            pos = pos.play_move(action, WHITE, False)
            color = BLACK
        elif (color == BLACK):
            action, probs = chooseActionAccordingToMCTS(pos, challengerPlayer.net_api)
            print("gewählte aktion ", action)
            pos = pos.play_move(action, BLACK, False)
            color = WHITE

    # update winner when game is finished for all experiences in this single game

    winner = pos.result()
    if winner == currentPlayer.color:
        return currentPlayer
    else:
        return challengerPlayer
    # print(pos.result())
    # print(pos.result_string())
    # replayBuffer.addToReplayBuffer(trainingSet)
    # return pos


def chooseActionAccordingToMCTS(pos, nn_api):
    initial_board_state = GoGamestate(pos.board, constants.board_size, pos.to_play, pos)

    root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state, move_from_parent=None,
                                                  parent=None)
    mcts = MonteCarloTreeSearch(root, nn_api)
    resultChild = mcts.search_function(constants.mcts_simulations)

    return coords.from_flat(resultChild.move_from_parent), root.getProbDistributionForChildren()


def randomStartPlayer():
    return 1 if random.random() < 0.5 else -1
    #return 1
