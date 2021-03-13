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
