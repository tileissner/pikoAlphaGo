from components.go.goEngineApi import createGame
import random
import components.go.coords as coords
from components.mcts.goMCTS import GoGamestate
from components.mcts.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from components.mcts.search import MonteCarloTreeSearch
from components.nn.nn_api import NetworkAPI
from components.player.player import Player
from utils import constants
import matplotlib.pyplot as plt

WHITE, NONE, BLACK = range(-1, 2)
currentNetFileName = "C:\\Users\\user\Documents\git\pikoAlphaGo\components\\nn\models\model20210119-145041"
nn_api = NetworkAPI()
nn_api.model_load(currentNetFileName)


def evaluateNet(board_size, netColor, currentNetFileName):
    pos = createGame(board_size, None)

    netWins = 0
    randomPlayerWins = 0
    draws = 0

    netPlayer = Player(netColor, currentNetFileName)
    randomPlayer = Player(netColor*-1, currentNetFileName)
    colorstring = "BLACK" if netColor is BLACK else "WHITE"
    randomcolorstring = "WHITE" if netColor is BLACK else "BLACK"

    for _ in range(0, constants.amount_evaluator_iterations):
        # Zufällige Auswahl wer beginnt
        winner, score = startGameEvaluation(pos, BLACK, netPlayer, randomPlayer)
        if winner is not None:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("BLACK" if winner.color == 1 else "WHITE", "won with a score of", score)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elif winner is None:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("DRAW!")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if winner == netPlayer:
            netWins += 1
        elif winner == randomPlayer:
            randomPlayerWins += 1
        elif winner == None:
            draws+=1
    print("Net won", netWins, "out of", constants.amount_evaluator_iterations, "games with", colorstring)
    print("Random player won", randomPlayerWins, "out of", constants.amount_evaluator_iterations, "games with", randomcolorstring)
    print(draws, "games were drawn.")
    createPlot(netWins,randomPlayerWins,draws)

def createPlot(netWins,randomPlayerwins,draws):
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

def startGameEvaluation(pos, color, netPlayer, randomPlayer):
    while not pos.is_game_over():
        if (color == WHITE):
            if(netPlayer.color==WHITE):
                coord, probdist = chooseActionAccordingToMCTS(pos, nn_api)
                pos = pos.play_move(coord)
                print("Net played: ", coord)
                color = BLACK
            else:
                validMoveList = pos.all_legal_moves().tolist()
                validMoveIndices = [i for i, e in enumerate(validMoveList) if e == 1]
                randomnumber = random.randint(0, len(validMoveIndices)-1)
                #print("Random: ", randomnumber)
                #print("Random move: ", validMoveIndices[randomnumber])
                coord = coords.from_flat(validMoveIndices[randomnumber])
                pos = pos.play_move(coord)
                print("Random player played: ", coord)
                color = BLACK

        elif (color == BLACK):
            if(netPlayer.color==BLACK):
                coord, probdist = chooseActionAccordingToMCTS(pos, nn_api)
                pos = pos.play_move(coord)
                print("Net played: ", coord)
                color = WHITE
            else:
                validMoveList = pos.all_legal_moves().tolist()
                validMoveIndices = [i for i, e in enumerate(validMoveList) if e == 1]
                randomnumber = random.randint(0, len(validMoveIndices)-1)
                #print("Random: ", randomnumber)
                #print("Random move: ", validMoveIndices[randomnumber])
                coord = coords.from_flat(validMoveIndices[randomnumber])
                pos = pos.play_move(coord)
                print("Random player played: ", coord)
                color = WHITE



    # update winner when game is finished for all experiences in this single game

    winner = pos.result()
    if winner == netPlayer.color:
        return netPlayer, pos.score()
    elif winner == 0:
        return None, pos.score()
    elif winner == randomPlayer.color:
        return randomPlayer, pos.score()

def chooseActionAccordingToMCTS(pos, nn_api):
    initial_board_state = GoGamestate(pos.board, constants.board_size, pos.to_play, pos)

    root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state, move_from_parent=None,
                                                  parent=None)
    mcts = MonteCarloTreeSearch(root, nn_api)
    resultChild = mcts.search_function(constants.mcts_simulations)

    return coords.from_flat(resultChild.move_from_parent), root.getProbDistributionForChildren()



evaluateNet(5,BLACK,currentNetFileName)