import random

import matplotlib.pyplot as plt

from components.go import coords
import utils.readConfigFile as configFile
from components.go.goEngineApi import createGame
from components.mcts.goMCTS import GoGamestate
from components.mcts.search import MonteCarloTreeSearch
from components.nn.nn_api import NetworkAPI
from components.player.player import Player
from utils import constants

'''
Klasse zum Testen der eigens erstellten Netzewrke gegen den reinzufällig spielenden Agenten
'''

WHITE, NONE, BLACK = range(-1, 2)
currentNetFileName = "/home/marcel/Downloads/model20210311-204904"
nn_api = NetworkAPI()
nn_api.model_load(currentNetFileName)

# read config file and store it in constants.py
constants.configFileLocation = "config.yaml"
configFile.readConfigFile(constants.configFileLocation)


def evaluateNet(board_size, netColor, currentNetFileName, startPlayerColor):

    netWins = 0
    randomPlayerWins = 0
    draws = 0

    netPlayer = Player(netColor, currentNetFileName)
    randomPlayer = Player(netColor * -1, currentNetFileName)
    colorstring = "BLACK" if netColor is BLACK else "WHITE"
    randomcolorstring = "WHITE" if netColor is BLACK else "BLACK"

    for _ in range(0, constants.amount_evaluator_iterations):
        # Zufällige Auswahl wer beginnt
        print("erstelle neues Spiel, Beginner " + str(startPlayerColor))
        pos = createGame(board_size, startPlayerColor)
        startPlayerColor = startPlayerColor * (-1) #TODO hier toggle für swap
        winner, score = startGameEvaluation(pos, netPlayer, randomPlayer)
        if winner is not None:
            print(
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("BLACK" if winner.color == 1 else "WHITE", "won with a score of", score)
            print(
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elif winner is None:
            print(
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("DRAW!")
            print(
                "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if winner == netPlayer:
            netWins += 1
        elif winner == randomPlayer:
            randomPlayerWins += 1
        elif winner == None:
            draws += 1
    print("Net won", netWins, "out of", constants.amount_evaluator_iterations, "games with", colorstring)
    print("Random player won", randomPlayerWins, "out of", constants.amount_evaluator_iterations, "games with",
          randomcolorstring)
    print(draws, "games were drawn.")
    createPlot(netWins, randomPlayerWins, draws)


def createPlot(netWins, randomPlayerwins, draws):
    if draws == 0:
        labels = 'Net wins', 'Random player wins'
        sizes = [netWins, randomPlayerwins]
    else:
        labels = 'Net wins', 'Random player wins', 'Draws'
        sizes = [netWins, randomPlayerwins, draws]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=None, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title("Winners of " + str(constants.amount_evaluator_iterations) + " games")
    plt.savefig("winnerdiagram.jpg")


def startGameEvaluation(pos, netPlayer, randomPlayer):
    initial_board_state = GoGamestate(pos.board, constants.board_size, pos)
    mcts = MonteCarloTreeSearch(nn_api, 0.0, initial_board_state)
    move_counter = 0
    color = pos.to_play
    while not pos.is_game_over() and move_counter < constants.board_size ** 2 * 2:
        if color == WHITE:
            if netPlayer.color == WHITE:
                root = mcts.search_mcts_function()
                action = root.select_action(temperature=0)
                pos = pos.play_move(coords.from_flat(action))
                print("Net played: ", action)
                print(pos)
                color = BLACK
            else:
                validMoveList = pos.all_legal_moves().tolist()
                validMoveIndices = [i for i, e in enumerate(validMoveList) if e == 1]
                randomnumber = random.randint(0, len(validMoveIndices) - 1)
                coord = coords.from_flat(validMoveIndices[randomnumber])
                pos = pos.play_move(coord)
                print("Random player played: ", coords.to_flat(coord))
                print(pos)
                color = BLACK

        elif color == BLACK:
            if netPlayer.color == BLACK:
                root = mcts.search_mcts_function()
                action = root.select_action(temperature=0)
                pos = pos.play_move(coords.from_flat(action))
                print("Net played: ", action)
                print(pos)
                color = WHITE
            else:
                validMoveList = pos.all_legal_moves().tolist()
                validMoveIndices = [i for i, e in enumerate(validMoveList) if e == 1]
                randomnumber = random.randint(0, len(validMoveIndices) - 1)
                coord = coords.from_flat(validMoveIndices[randomnumber])
                pos = pos.play_move(coord)
                print("Random player played: ", coord)
                print(pos)
                color = WHITE

        mcts.go_game_state = GoGamestate(pos.board, constants.board_size, pos)
        move_counter = move_counter + 1


    winner = pos.result()
    if winner == netPlayer.color:
        return netPlayer, pos.score()
    elif winner == 0:
        return None, pos.score()
    elif winner == randomPlayer.color:
        return randomPlayer, pos.score()


# net player = black
evaluateNet(constants.board_size, BLACK, currentNetFileName, BLACK)
