import math
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

WHITE, NONE, BLACK = range(-1, 2)


def evaluateNet(board_size, color, currentNetFileName, challengerNetFileName):
    pos = createGame(board_size, color)
    currentPlayerWins = 0
    challengerPlayerWins = 0

    # NEVER CHANGE COLOR
    currentPlayer = Player(WHITE, currentNetFileName)
    challengerPlayer = Player(BLACK, challengerNetFileName)

    for _ in range(0, constants.amount_evaluator_iterations):
        # Zufällige Auswahl wer beginnt


        winner = startGameEvaluation(pos, currentPlayer, challengerPlayer)
        if winner == currentPlayer:
            currentPlayerWins += 1
        else:
            challengerPlayerWins += 1

    print(str(currentPlayer.color) + " hat " + str(currentPlayerWins) + " wins")
    #print(currentPlayer.color, " hat ", currentPlayerWins, " wins")
    print(str(challengerPlayer.color) + " hat " + str(challengerPlayerWins) + " wins")

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
                      board_deltas=None, to_play=BLACK)
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
        return "WHITE"
    elif color == 1:
        return "BLACK"
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



    initial_board_state = GoGamestate(pos.board, constants.board_size, pos.to_play, pos)

    root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state, move_from_parent=None,
                                                  parent=None)

    mcts = MonteCarloTreeSearch(root, net_api, (1/math.sqrt(2)))
    # return coords.from_flat(resultChild.move_from_parent), root.getProbDistributionForChildren(),

    move_counter = 0

    while not pos.is_game_over() and move_counter < constants.board_size **2 *2:

        # print(pos.board)
        # resultChild = mcts.search_function(constants.mcts_simulations)
        #new_root = mcts.search_function(constants.mcts_simulations)
        new_root = mcts._new_search_function(constants.mcts_simulations)


        action = coords.from_flat(new_root.move_from_parent)
        probs = new_root.parent.getProbDistributionForChildren()

        new_root = discard_tree(new_root)

        root = new_root
        mcts.root = new_root
        #action, probs = chooseActionAccordingToMCTS(pos, net_api, root)

        #GetMoveProbablitiesFrom MCTS


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


        print("gewählte aktion {} von {}".format(action, color))
        # print(pos)
        print(pos.board)

        newTrainingSet = TrainingSet(pos.board, probs, currentColor)
        trainingSet.append(newTrainingSet)
        move_counter = move_counter + 1


    print("gespielte züge {}".format(move_counter))

    # update winner when game is finished for all experiences in this single game
    for newTrainingSet in trainingSet:
        newTrainingSet.updateWinner(pos.result())

    # print(pos.result())
    # print(pos.result_string())
    # replayBuffer.addToReplayBuffer(trainingSet)
    # return pos
    return trainingSet


def startGameEvaluation(pos, currentPlayer, challengerPlayer):

    color = None
    if randomStartPlayer() == currentPlayer.color:
        color = currentPlayer.color

    else:
        color = challengerPlayer.color

    initial_board_state = GoGamestate(pos.board, constants.board_size, pos.to_play, pos)


    current_player_new_root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state, move_from_parent=None,
                                                                     parent=None)
    challengerPlayer_player_new_root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state,
                                                                              move_from_parent=None,
                                                                              parent=None)

    currentPlayer.mcts = MonteCarloTreeSearch(current_player_new_root, currentPlayer.net_api, 0.)
    challengerPlayer.mcts = MonteCarloTreeSearch(challengerPlayer_player_new_root, challengerPlayer.net_api, 0.)

    challengerPlayer_player_new_root.expand()
    current_player_new_root.expand()



    while not pos.is_game_over():

        # current_player_new_root = mcts_current_player.search_function(constants.mcts_simulations)
        # challenger_player_new_root = mcts_challenger_player.search_function(constants.mcts_simulations)

        # print(str(color) + " (" + getPlayerName(color) + ") am Zug")v

        # print("Random Number: " + str(randomNum))

        if (color == WHITE):
            #action, probs = chooseActionAccordingToMCTS(pos, currentPlayer.net_api)
            current_player_new_root = currentPlayer.mcts.search_function(constants.mcts_simulations)
            action = coords.from_flat(current_player_new_root.move_from_parent)
            #TODO Root des anderen spieler setzen
            # in der evaluation kein trainingsset -> probs nicht benötigt
            #probs = current_player_new_root.parent.getProbDistributionForChildren()

            print("gewählte aktion von weiß ", action)
            pos = pos.play_move(action, WHITE, False)
            for child_node in challengerPlayer_player_new_root.children:
                if child_node.move_from_parent == action:
                    challengerPlayer_player_new_root = child_node


            currentPlayer.mcts.root = current_player_new_root
            color = BLACK
        elif (color == BLACK):
            #action, probs = chooseActionAccordingToMCTS(pos, challengerPlayer.net_api)
            challengerPlayer_player_new_root = challengerPlayer.mcts.search_function(constants.mcts_simulations)
            action = coords.from_flat(challengerPlayer_player_new_root.move_from_parent)
            # TODO Root des anderen spieler setzen
            print("gewählte aktion von schwarz ", action)
            pos = pos.play_move(action, BLACK, False)

            for child_node in current_player_new_root.children:
                if child_node.move_from_parent == action:
                    current_player_new_root = child_node

            challengerPlayer.mcts.root = challengerPlayer_player_new_root
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


def chooseActionAccordingToMCTS(pos, nn_api, root):


    mcts = MonteCarloTreeSearch(root, nn_api)
    resultChild = mcts.search_function(constants.mcts_simulations)


    # for node in aktuellem node:
    #     wenn nicht resultChild:
    #         lösche node

    return resultChild
    # return coords.from_flat(resultChild.move_from_parent), root.getProbDistributionForChildren(),


def randomStartPlayer():
    return 1 if random.random() < 0.5 else -1
    #return 1

def discard_tree(node):
    """
    keep best child as root and remove everything that is not a child of the given node
    """
    if node.parent is not None:
        node.parent = None

    return node