import random
import threading

import numpy as np

import components.go.coords as coords
import components.go.go as go
from components.go.trainingSet import TrainingSet


from components.mcts.goMCTS import GoGamestate
from components.mcts.search import MonteCarloTreeSearch
from components.nn.nn_api import NetworkAPI
from components.player.player import Player
from utils import constants

WHITE, NONE, BLACK = range(-1, 2)


def evaluateNet(board_size, color, currentNetFileName, challengerNetFileName, thread_counter):


    currentPlayerWins = 0
    challengerPlayerWins = 0

    # NEVER CHANGE COLOR
    currentPlayer = Player(WHITE, currentNetFileName)
    challengerPlayer = Player(BLACK, challengerNetFileName)

    for _ in range(0, constants.games_per_eval_thread):
        pos = createGame(board_size, color)

        # Zufällige Auswahl wer beginnt

        winner = startGameEvaluation(pos, currentPlayer, challengerPlayer, thread_counter)
        if winner == currentPlayer:
            constants.current_player_wins += 1
        else:
            constants.challenger_wins += 1
        color = color * (-1)

    return currentPlayerWins, challengerPlayerWins




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


# def getMockProbabilities(pos):
#     # probabilities = {}
#     probabilities = []
#
#     # Performance Update: Without if statement?
#
#     # for i in range(0, pos.all_legal_moves().size):
#     index = 0
#     for move in pos.all_legal_moves():
#         if move == 1:
#             # probabilities.update({index: random.random()})
#             if index == (constants.board_size * constants.board_size):
#                 probabilities.append(0.00001)
#             else:
#                 probabilities.append(random.random())
#         else:
#             # probabilities.update({index: 0.0})
#             probabilities.append(0.0)
#         index += 1
#
#     return probabilities


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

    initial_board_state = GoGamestate(pos.board, constants.board_size, pos)

    # root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state, move_from_parent=None,
    #                                               parent=None)

    # mcts = MonteCarloTreeSearch(root, net_api, (1/math.sqrt(2)))
    mcts = MonteCarloTreeSearch(net_api, constants.c_puct, initial_board_state)
    # return coords.from_flat(resultChild.move_from_parent), root.getProbDistributionForChildren(),

    move_counter = 0
    print("spiel startet, beginner: {}".format(pos.to_play))
    new_root = None

    while not pos.is_game_over() and move_counter < constants.board_size ** 2 * 2:

        # print(pos.board)
        # resultChild = mcts.search_function(constants.mcts_simulations)
        # new_root = mcts.search_function(constants.mcts_simulations)
        # new_root = mcts._new_search_function(constants.mcts_simulations)
        root = mcts.search_mcts_function(new_root)

        action_probs = [0 for _ in range(constants.board_size * constants.board_size + 1)]
        for k, v in root.children.items():
            action_probs[k] = v.visit_count

        action_probs = action_probs / np.sum(action_probs)

        action = root.select_action(temperature=constants.temperature)
        pos = pos.play_move(coords.from_flat(action))
        mcts.go_game_state = GoGamestate(pos.board, constants.board_size, pos)
        print("move counter {}".format(move_counter))
        print("chosen action {} by {}".format(action, pos.to_play * (-1)))
        # print(pos)
        # print(pos.board)
        new_root = root.children[action]

        newTrainingSet = TrainingSet(pos.board, action_probs, pos.to_play)
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


def startGameEvaluation(pos, currentPlayer, challengerPlayer, thread_counter):
    c_puct = constants.c_puct + random.uniform(-1.0, 1.0)
    print("thread {} hat c_puct {}".format(thread_counter, c_puct))
    # currentPlayer = WHITE = -1
    # challengerPlayer = BLACK = 1


    initial_board_state = GoGamestate(pos.board, constants.board_size, pos)

    #
    # current_player_new_root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state, move_from_parent=None,
    #                                                                  parent=None)
    # challengerPlayer_player_new_root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state,
    #                                                                           move_from_parent=None,
    #                                                                           parent=None)

    # currentPlayer.mcts = MonteCarloTreeSearch(current_player_new_root, currentPlayer.net_api, 0.)
    # challengerPlayer.mcts = MonteCarloTreeSearch(challengerPlayer_player_new_root, challengerPlayer.net_api, 0.)
    currentPlayer.mcts = MonteCarloTreeSearch(currentPlayer.net_api, c_puct, initial_board_state)
    challengerPlayer.mcts = MonteCarloTreeSearch(challengerPlayer.net_api, c_puct, initial_board_state)

    # TODO needed?
    # challengerPlayer_player_new_root.expand()
    # current_player_new_root.expand()

    move_counter = 0

    challenger_player_new_root = None
    current_player_new_root = None

    while not pos.is_game_over() and move_counter < constants.board_size ** 2 * 2:

        # current_player_new_root = mcts_current_player.search_function(constants.mcts_simulations)
        # challenger_player_new_root = mcts_challenger_player.search_function(constants.mcts_simulations)

        # print(str(color) + " (" + getPlayerName(color) + ") am Zug")v

        # print("Random Number: " + str(randomNum))

        if pos.to_play == WHITE:
            # current_player_new_root = currentPlayer.mcts._new_search_function(constants.mcts_simulations)
            # action = current_player_new_root.move_from_parent
            # #TODO Root des anderen spieler setzen
            # # in der evaluation kein trainingsset -> probs nicht benötigt
            # #probs = current_player_new_root.parent.getProbDistributionForChildren()
            #
            #
            # pos = pos.play_move(coords.from_flat(action), WHITE, False)
            # for child_node in challengerPlayer_player_new_root.children:
            #     if child_node.move_from_parent == action:
            #         challengerPlayer_player_new_root = child_node
            #
            # currentPlayer.mcts.root = current_player_new_root
            # currentPlayer.mcts.root = discard_tree(currentPlayer.mcts.root)

            # neue mcts
            root = currentPlayer.mcts.search_mcts_function(current_player_new_root)

            action = root.select_action(temperature=0.01)
            pos = pos.play_move(coords.from_flat(action))
            synchronized_go_game_state = GoGamestate(pos.board, constants.board_size, pos)
            currentPlayer.mcts.go_game_state = synchronized_go_game_state
            challengerPlayer.mcts.go_game_state = synchronized_go_game_state

            current_player_new_root = root.children[action]
            if challenger_player_new_root is not None:
                challenger_player_new_root = challenger_player_new_root.children[action]

            print("white played ", action)
            print(pos.board)

        # elif (color == BLACK):
        else:
            # challengerPlayer_player_new_root = challengerPlayer.mcts._new_search_function(constants.mcts_simulations)
            # action = challengerPlayer_player_new_root.move_from_parent
            # # TODO Root des anderen spieler setzen
            #
            # pos = pos.play_move(coords.from_flat(action), BLACK, False)
            #
            # for child_node in current_player_new_root.children:
            #     if child_node.move_from_parent == action:
            #         current_player_new_root = child_node
            #
            # challengerPlayer.mcts.root = challengerPlayer_player_new_root
            # challengerPlayer.mcts.root = discard_tree(challengerPlayer.mcts.root)

            # neue mcts
            root = challengerPlayer.mcts.search_mcts_function(challenger_player_new_root)

            action = root.select_action(temperature=0.01)
            pos = pos.play_move(coords.from_flat(action))
            synchronized_go_game_state = GoGamestate(pos.board, constants.board_size, pos)
            currentPlayer.mcts.go_game_state = synchronized_go_game_state
            challengerPlayer.mcts.go_game_state = synchronized_go_game_state

            challenger_player_new_root = root.children[action]
            if current_player_new_root is not None:
                current_player_new_root = current_player_new_root.children[action]

            print("black played ", action)
            print(pos.board)

        move_counter += 1

    # update winner when game is finished for all experiences in this single game

    winner = pos.result()
    # if winner == currentPlayer.color:
    #     print(str(currentPlayer.color) + " has won")
    #     return currentPlayer
    # else:
    #     print(str(challengerPlayer.color) + " has won")
    #     return challengerPlayer
    if winner == currentPlayer.color:
        print(str(currentPlayer.color) + " has won")
        return currentPlayer
    elif winner == challengerPlayer.color:
        print(str(challengerPlayer.color) + " has won")
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
    # return 1


def discard_tree(node):
    """
    keep best child as root and remove everything that is not a child of the given node
    """
    if node.parent is not None:
        node.parent = None

    return node
