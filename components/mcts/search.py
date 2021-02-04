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

        #self.root = node

        self.net_api = net_api
        self.c_puct = c_puct
        self.go_game_state = go_game_state

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

    # def _tree_policy(self):
    #     """
    #     node selection
    #     selects node to run rollout/playout for
    #
    #     Returns
    #     -------
    #
    #     """
    #     current_node = self.root
    #
    #     while not current_node.is_terminal_node():
    #
    #         # aktuelle knoten nehmen
    #         # 1. prüfen ob er ncoh nicht besucht uwrde
    #         #     1.1 NN um hilfe fragen -> P[s][a] und v
    #         # 2. falls nicht -> alle kinder durchgehen "probieren"
    #         #     2.1 und dann returnen (? stimmt das so ?)
    #         # 3. neuer knoten = bestes kind
    #         # 4. nächste schleifeniteration
    #
    #         # step 1
    #         if current_node.n == 0:
    #             # print(net_api.getPredictionFromNN(current_node.state.board))
    #
    #             parentStates = []
    #             currentNodeCopy = deepcopy(current_node)
    #
    #
    #
    #             for i in range(0, constants.state_history_length):
    #                 if currentNodeCopy.parent is not None:
    #                     currentNodeCopy = deepcopy(currentNodeCopy.parent)
    #                     all_zeros = not np.any(currentNodeCopy.state.board)
    #                     equalToParent = np.array_equal(currentNodeCopy.state.board, current_node.state.board)
    #                     if not all_zeros and not equalToParent:
    #                         parentStates.append(currentNodeCopy.state.board)
    #
    #
    #             current_node.winner, current_node.p_distr = self.net_api.getPredictionFromNN(current_node.state.board,
    #                                                                                          parentStates,
    #                                                                                          current_node.state.pos.to_play)
    #             # Set illegal moves to 0
    #             current_node.p_distr = goEngineApi.zero_illegal_moves_from_prediction(current_node.state.pos,
    #                                                                                   current_node.p_distr)
    #
    #
    #             # if current_node.winner < 0:
    #             #     current_node.winner = -1
    #             # else:
    #             #     current_node.winner = 1
    #
    #             # current_node.p_distr = goEngineApi.getSemiMockProbabilities(current_node.state.pos,
    #             #                                                           current_node.p_distr)
    #             current_node._number_of_visits = 1
    #             return current_node
    #
    #         # step 2
    #         if not current_node.is_fully_expanded():
    #
    #             while not current_node.is_fully_expanded():
    #                 current_node.expand()  # Setzen von q,n = 0
    #
    #             return current_node
    #
    #         # step 3 + 4
    #
    #         current_node = current_node.best_child(self.c_puct)  # best_child geht den schritt in das beste kind
    #
    #         # -- ALTER PART --
    #         # # if self.node not in self.visitedNodes:
    #         # #     self.visitedNodes.append(current_node)
    #         # #     return current_node
    #         # if current_node.n == 0:
    #         #     return current_node
    #         #
    #         # #ausprobieren der kinder
    #         # if not current_node.is_fully_expanded():
    #         #     return current_node.expand()
    #         # else:
    #         #     current_node = current_node.best_child()
    #
    #     # außerhalb der while schleife
    #     if current_node.is_terminal_node():
    #         current_node.winner = current_node.state.game_result
    #
    #         current_node.p_distr = goEngineApi.getMockProbabilities(current_node.state.pos)  # von NN
    #
    #     return current_node

    def search_mcts_function(self, new_root):
        mcts = MCTS(self.go_game_state, self.net_api, self.c_puct)
        return mcts.run(self.net_api, self.go_game_state, self.go_game_state.pos.to_play, new_root)

    def _new_search_function(self, simulations_number):
        """
        node selection
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        iter = 0
        for _ in range(0, simulations_number):

            # Cheat sheet = https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0

            iter += 1

            parent_nodes = self.createHistoryStates(current_node)

            # Cheat sheet 1. step
            # descend to leaf node (using best_child method) until as long as we have children
            # SELECT PHASE
            while len(current_node.children) > 0:
                action = current_node.new_best_child(self.c_puct)
                #current_node = current_node.new_best_child(self.c_puct)
                already_exists_in_tree = False
                for child in current_node.children:
                    if child.move_from_parent == action:
                        current_node = child
                        already_exists_in_tree = True
                # if node is not connected to our tree yet, expand it
                if not already_exists_in_tree:
                    current_node = current_node.expand_specific_child(action)

            # Cheat sheet 2. step
            # predicted_winner = v of the cheat sheet
            # p_distr = p of the cheat sheet
            predicted_winner, current_node.p_distr = self.net_api.getPredictionFromNN(current_node.state.board,
                                                                                         parent_nodes,
                                                                                         current_node.state.pos.to_play)

            #while not current_node.is_fully_expanded():
            if not current_node.is_fully_expanded():
                current_node.expand()  # Setzen von q,n = 0


            # Cheat sheet 3. step
            current_node.backpropagate(predicted_winner)
            #current_node = self.root




        probs = self.getProbDistributionForChildren()
        # TODO: alternative: höchster q wert

        # exploration
        if constants.competitive == 0:
            probs = self.getProbDistributionForChildrenInExplorationMode()
            max_prob = 0.
            index = 0
            best_action = -1
            for prob in probs:
                if prob > max_prob:
                    max_prob = prob
                    best_action = index
                index += 1

            return best_action, probs

            # highest_value = -1
            # best_child = None
            # # TODO:
            # # bsp: 66% und 33%
            # # wahrscheinlichkeitsgenerator -> zwischen 0-66% -> dieser zug, andernfalls der andere
            # # für t = 1
            # for child in self.root.children:
            #     n_temperature = pow(child._number_of_visits, (1 / constants.temperature))
            #     if n_temperature > highest_value:
            #         highest_value = n_temperature
            #         best_child = child


        # exploitation
        else:
            highest_visit_count = -1
            best_child = None
            for child in self.root.children:
                if child._number_of_visits > highest_visit_count:
                    highest_visit_count = child._number_of_visits
                    best_child = child

        #action wäre best_child.move_from_parent

        # return self.root #sollte keinen sinn ergeben oder?
        return best_child



    def getProbDistributionForChildrenInExplorationMode(self):
        number_of_visits_for_children = [0.] * 26


        #Iterate over children and get move from parent
        for c in self.root:
            index_in_array = c.move_from_parent
            number_of_visits_for_children[index_in_array] = pow(c._number_of_visits, (1/constants.temperature))


        probs_from_mcts = [number_of_visits_for_children[i]/sum(number_of_visits_for_children) for i in range(len(number_of_visits_for_children))]
        return probs_from_mcts

    def getProbDistributionForChildrenInCompetitiveMode(self):
        number_of_visits_for_children = [0.] * 26


        #Iterate over children and get move from parent
        for c in self.root:
            children_visit_count = c._number_of_visits
            index_in_array = c.move_from_parent
            number_of_visits_for_children[index_in_array] = children_visit_count


        probs_from_mcts = [number_of_visits_for_children[i]/sum(number_of_visits_for_children) for i in range(len(number_of_visits_for_children))]
        return probs_from_mcts

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