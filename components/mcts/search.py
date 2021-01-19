from random import random

import numpy as np

from components.go import goEngineApi
from utils import constants
from copy import deepcopy


class MonteCarloTreeSearch(object):

    def __init__(self, node, net_api):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """

        self.root = node

        self.net_api = net_api

    # ehemals best_action
    def search_function(self, simulations_number):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------

        """
        # TODO: Fall Terminal Node abfangen
        for _ in range(0, simulations_number):
            v = self._tree_policy()

            v.backpropagate(v.winner)

        return self.root.best_child(c_puct=0.)

    def _tree_policy(self):
        """
        node selection
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root

        while not current_node.is_terminal_node():

            # aktuelle knoten nehmen
            # 1. prüfen ob er ncoh nicht besucht uwrde
            #     1.1 NN um hilfe fragen -> P[s][a] und v
            # 2. falls nicht -> alle kinder durchgehen "probieren"
            #     2.1 und dann returnen (? stimmt das so ?)
            # 3. neuer knoten = bestes kind
            # 4. nächste schleifeniteration

            # step 1
            if current_node.n == 0:
                # print(net_api.getPredictionFromNN(current_node.state.board))

                parentStates = []
                currentNodeCopy = deepcopy(current_node)


                for i in range(0, constants.state_history_length):
                    if currentNodeCopy.parent is not None:
                        currentNodeCopy = deepcopy(currentNodeCopy.parent)
                        all_zeros = not np.any(currentNodeCopy.state.board)
                        equalToParent = np.array_equal(currentNodeCopy.state.board, current_node.state.board)
                        if not all_zeros and not equalToParent:
                            parentStates.append(currentNodeCopy.state.board)

                current_node.winner, current_node.p_distr = self.net_api.getPredictionFromNN(current_node.state.board,
                                                                                             parentStates,
                                                                                             current_node.state.pos.to_play)
                # Set illegal moves to 0
                current_node.p_distr = goEngineApi.zero_illegal_moves_from_prediction(current_node.state.pos,
                                                                                      current_node.p_distr)

                # TODO muss mit richtigen werten ersetzt werden
                # if current_node.winner < 0:
                #     current_node.winner = -1
                # else:
                #     current_node.winner = 1

                # current_node.p_distr = goEngineApi.getSemiMockProbabilities(current_node.state.pos,
                #                                                           current_node.p_distr)

                return current_node

            # step 2
            if not current_node.is_fully_expanded():

                while not current_node.is_fully_expanded():
                    current_node.expand()  # Setzen von q,n = 0

                return current_node

            # step 3 + 4
            # TODO passt das mit c_puct?
            current_node = current_node.best_child(c_puct=4.)  # best_child geht den schritt in das beste kind

            # -- ALTER PART --
            # # if self.node not in self.visitedNodes:
            # #     self.visitedNodes.append(current_node)
            # #     return current_node
            # if current_node.n == 0:
            #     return current_node
            #
            # #ausprobieren der kinder
            # if not current_node.is_fully_expanded():
            #     return current_node.expand()
            # else:
            #     current_node = current_node.best_child()

        # außerhalb der while schleife
        if current_node.is_terminal_node():
            current_node.winner = current_node.state.game_result
            #TODO sinnvoll? lieber 0en?
            current_node.p_distr = goEngineApi.getMockProbabilities(current_node.state.pos)  # von NN

        return current_node

    def randomWinner(self):
        return 1 if random() < 0.5 else -1
