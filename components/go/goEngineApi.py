import random
import threading

import numpy as np

#import components.go.coords as coords
from components.go import coords
import components.go.go as go
from components.go.trainingSet import TrainingSet


from components.mcts.goMCTS import GoGamestate
from components.mcts.search import MonteCarloTreeSearch
from components.nn.nn_api import NetworkAPI
from components.player.player import Player
from utils import constants

WHITE, NONE, BLACK = range(-1, 2)


def evaluateNet(board_size, color, currentNetFileName, challengerNetFileName, thread_counter):

    # NEVER CHANGE COLOR
    currentPlayer = Player(WHITE, currentNetFileName)
    print("erstelle current player (WHITE) mit model " + str(currentNetFileName))
    challengerPlayer = Player(BLACK, challengerNetFileName)
    print("erstelle challenger player (BLACK) mit model " + str(challengerNetFileName))

    for _ in range(0, constants.games_per_eval_thread):
        pos = createGame(board_size, color)

        # Zufällige Auswahl wer beginnt

        winner = startGameEvaluation(pos, currentPlayer, challengerPlayer, thread_counter)
        if winner == currentPlayer:
            constants.current_player_wins += 1
        elif winner == challengerPlayer:
            constants.challenger_wins += 1
        else:
            constants.draws += 1
        color = color * (-1)





def selfPlay(board_size, color):
    pos = createGame(board_size, color)
    # trainingSet = startGame(pos, color)
    # return startGame(pos, color)
    return startGameMCTS(pos)


def createGame(N, beginner):
    EMPTY_BOARD = np.zeros([N, N], dtype=np.int8)
    pos = go.Position(EMPTY_BOARD, n=0, komi=0.0, caps=(0, 0),
                      lib_tracker=None, ko=None, recent=tuple(),
                      board_deltas=None, to_play=beginner)
    return pos


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


def startGameMCTS(pos):
    trainingSet = []

    # Temoporärer Ladevorgang d Netzes

    net_api = NetworkAPI()
    net_api.model_load(constants.currentBestNetFileName)

    initial_board_state = GoGamestate(pos.board, constants.board_size, pos)
    mcts = MonteCarloTreeSearch(net_api, constants.c_puct, initial_board_state)

    move_counter = 0
    print("spiel startet, beginner: {}".format(pos.to_play))

    while not pos.is_game_over() and move_counter < constants.board_size ** 2 * 2:

        root = mcts.search_mcts_function()

        action_probs = [0 for _ in range(constants.board_size * constants.board_size + 1)]
        for k, v in root.children.items():
            action_probs[k] = v.visit_count

        action_probs = action_probs / np.sum(action_probs)

        action = root.select_action(temperature=constants.temperature)
        pos = pos.play_move(coords.from_flat(action))
        mcts.go_game_state = GoGamestate(pos.board, constants.board_size, pos)
        print("move counter {}".format(move_counter))
        print("chosen action {} by {}".format(action, pos.to_play * (-1)))

        newTrainingSet = TrainingSet(pos.board, action_probs, pos.to_play)
        trainingSet.append(newTrainingSet)
        move_counter = move_counter + 1

    print("gespielte züge {}".format(move_counter))

    # update winner when game is finished for all experiences in this single game
    for newTrainingSet in trainingSet:
        newTrainingSet.updateWinner(pos.result())

    return trainingSet


def startGameEvaluation(pos, currentPlayer, challengerPlayer, thread_counter):
    c_puct = constants.c_puct + random.uniform(-1.0, 1.0)
    print("thread {} hat c_puct {}".format(thread_counter, c_puct))
    # currentPlayer = WHITE = -1
    # challengerPlayer = BLACK = 1


    initial_board_state = GoGamestate(pos.board, constants.board_size, pos)

    currentPlayer.mcts = MonteCarloTreeSearch(currentPlayer.net_api, c_puct, initial_board_state)
    challengerPlayer.mcts = MonteCarloTreeSearch(challengerPlayer.net_api, c_puct, initial_board_state)

    move_counter = 0

    while not pos.is_game_over() and move_counter < constants.board_size ** 2 * 2:

        if pos.to_play == WHITE:
            root = currentPlayer.mcts.search_mcts_function()

            action = root.select_action(temperature=0.)
            pos = pos.play_move(coords.from_flat(action))
            synchronized_go_game_state = GoGamestate(pos.board, constants.board_size, pos)
            currentPlayer.mcts.go_game_state = synchronized_go_game_state
            challengerPlayer.mcts.go_game_state = synchronized_go_game_state

            print("white played ", action)
            print(pos.board)

        else:
            root = challengerPlayer.mcts.search_mcts_function()

            action = root.select_action(temperature=0.)
            pos = pos.play_move(coords.from_flat(action))
            synchronized_go_game_state = GoGamestate(pos.board, constants.board_size, pos)
            currentPlayer.mcts.go_game_state = synchronized_go_game_state
            challengerPlayer.mcts.go_game_state = synchronized_go_game_state

            print("black played ", action)
            print(pos.board)

        move_counter += 1

    winner = pos.result()
    if winner == currentPlayer.color:
        print(str(currentPlayer.color) + " has won")
        return currentPlayer
    elif winner == challengerPlayer.color:
        print(str(challengerPlayer.color) + " has won")
        return challengerPlayer
    else:
        return None

def randomStartPlayer():
    return 1 if random.random() < 0.5 else -1
    # return 1


def discard_tree(node):
    """
    keep best child as root and remove everything that is not a child of the given node
    """
    if node.parent is not None:
        node.parent = None

    return node
