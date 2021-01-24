from random import random

import numpy as np

from components.go import goEngineApi
from utils import constants
from copy import deepcopy


class MonteCarloTreeSearch(object):

    def __init__(self, node, net_api, c_puct):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """

        self.root = node

        self.net_api = net_api
        self.c_puct = c_puct

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
        count = 0
        for _ in range(0, simulations_number):
            v = self._tree_policy()

            if v._number_of_visits == 0:
                v._number_of_visits = 1

            if v.parent is not None:
                v.parent.backpropagate(v.winner)

            #v.backpropagate(v.winner)
            count += 1
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
                current_node._number_of_visits = 1
                return current_node

            # step 2
            if not current_node.is_fully_expanded():

                while not current_node.is_fully_expanded():
                    current_node.expand()  # Setzen von q,n = 0

                return current_node

            # step 3 + 4
            # TODO passt das mit c_puct?
            current_node = current_node.best_child(self.c_puct)  # best_child geht den schritt in das beste kind

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


    def _new_search_function(self, simulations_number):
        """
        node selection
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root


        for _ in range(0, simulations_number):
            # bestchild aufgerufen bis leaf node -- SOlange es kinder gibt
            while not len(current_node.children) == 0:
                current_node = current_node.best_child(self.c_puct)

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
            #TODO: IF NICHT benötigt?
            if not current_node.is_fully_expanded():
                while not current_node.is_fully_expanded():
                    current_node.expand()  # Setzen von q,n = 0

            current_node.backpropagate()

            #TODO: Überprüfung in while schleife?

            # if current_node.is_terminal_node():
            #     current_node.winner = current_node.state.game_result
            #     # TODO sinnvoll? lieber 0en?
            #     current_node.p_distr = goEngineApi.getMockProbabilities(current_node.state.pos)  # von NN


        return self.root

        #Kinder vorhanden?
        # Ja --> Bestchild
        # Nein --> Expandiere alle Kinder
            # BP vom Knoten dessen Kinder gerade Expandiert wurden

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
                current_node._number_of_visits = 1
                return current_node

            # step 2
            if not current_node.is_fully_expanded():

                while not current_node.is_fully_expanded():
                    current_node.expand()  # Setzen von q,n = 0

                return current_node

            # step 3 + 4
            # TODO passt das mit c_puct?
            current_node = current_node.best_child(self.c_puct)  # best_child geht den schritt in das beste kind

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
