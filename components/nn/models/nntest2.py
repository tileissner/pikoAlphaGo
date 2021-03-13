#%%
import random

import numpy as np

import utils.readConfigFile as configFile
from components.go import coords
from components.go.coords import from_flat
from components.go.goEngineApi import createGame
from components.mcts.goMCTS import GoGamestate
from components.mcts.search import MonteCarloTreeSearch
from components.nn.nn_api import NetworkAPI
from components.player.player import Player
from utils import constants
from split_input import split_input



constants.configFileLocation = "/home/marcel/PycharmProjects/pikoAlphaGo/config.yaml"
# read config file and store it in constants.py
configFile.readConfigFile(constants.configFileLocation)


WHITE, NONE, BLACK = range(-1, 2)


pos = createGame(5, BLACK)
c_puct = constants.c_puct + random.uniform(-1.0, 1.0)
pos = pos.play_move(from_flat(14))#b
pos = pos.play_move(from_flat(0))#w
pos = pos.play_move(from_flat(7))#b
pos = pos.play_move(from_flat(10))#w
pos = pos.play_move(from_flat(8))#b
pos = pos.play_move(from_flat(17))#w





print(pos.board)

print("thread {} hat c_puct {}".format(1, c_puct))
# currentPlayer = WHITE = -1
# challengerPlayer = BLACK = 1


initial_board_state = GoGamestate(pos.board, constants.board_size, pos)

challengerPlayer = Player(BLACK, "models/model20210311-204904")

challengerPlayer.mcts = MonteCarloTreeSearch(challengerPlayer.net_api, c_puct, initial_board_state)

move_counter = 0


root = challengerPlayer.mcts.search_mcts_function()

action = root.select_action(temperature=0.)
pos = pos.play_move(coords.from_flat(action))
synchronized_go_game_state = GoGamestate(pos.board, constants.board_size, pos)
challengerPlayer.mcts.go_game_state = synchronized_go_game_state
