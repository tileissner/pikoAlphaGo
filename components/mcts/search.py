from random import random

import numpy as np

from components.mcts.node import MCTS
from utils import constants


class MonteCarloTreeSearch(object):

    #def __init__(self, node, net_api, c_puct, go_game_state):
    def __init__(self, net_api, c_puct, go_game_state):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """

        self.net_api = net_api
        self.c_puct = c_puct
        self.go_game_state = go_game_state

    def search_mcts_function(self):
        mcts = MCTS(self.go_game_state, self.net_api, self.c_puct)
        return mcts.run(self.net_api, self.go_game_state, self.go_game_state.pos.to_play)


    def randomWinner(self):
        return 1 if random() < 0.5 else -1


    def createHistoryStates(self, current_node):
        parentStates = []
        current_node.state.pos

        state_t_1 = None
        state_t_2 = None
        state_t_3 = None

        if len(current_node.state.pos.board_deltas) > 0:
            state_t_1 = current_node.state.pos.board - current_node.state.pos.board_deltas[0]
        if len(current_node.state.pos.board_deltas) > 1:
            state_t_2 = state_t_1 - current_node.state.pos.board_deltas[1]
             # state_t_2 = current_node.state.pos-current_node.state.pos.board_deltas[0]-current_node.state.pos.board_deltas[1] #equivalent
        if len(current_node.state.pos.board_deltas) > 2:
            state_t_3 = state_t_2 - current_node.state.pos.board_deltas[2]

        if state_t_1 is None:
            state_t_1 = np.zeros([constants.board_size, constants.board_size], dtype=np.int8)
        if state_t_2 is None:
            state_t_2 = np.zeros([constants.board_size, constants.board_size], dtype=np.int8)
        if state_t_3 is None:
            state_t_3 = np.zeros([constants.board_size, constants.board_size], dtype=np.int8)

        parentStates.append(state_t_1)
        parentStates.append(state_t_2)
        parentStates.append(state_t_3)
        return parentStates