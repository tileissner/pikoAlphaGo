import math

import numpy as np

from utils import constants


def uct_score(parent, child, c_puct):
    """
    The score for an action that would transition between the parent and child.
    """
    # prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    # if child.visit_count > 0:
    #     # The value of the child is from the perspective of the opposing player
    #     value_score = -child.value()
    # else:
    #     value_score = 0

    children_visit_count = 0
    for key, value in parent.children.items():
        children_visit_count += value.visit_count

    q_value = child.value()
    u_value = c_puct * child.prior * (math.sqrt(parent.visit_count) / (1 + children_visit_count))

    # q_value = value
    # # u_value = c_puct * self.p_distr[c.move_from_parent] * math.sqrt((children_visit_count) / (1 + self.n))
    # u_value = c_puct * self.p_distr[key] * (math.sqrt(self.n) / (1 + children_visit_count))

    return q_value + u_value


class Node:
    def __init__(self, prior, to_play, c_puct):
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.go_game_state = None
        self.c_puct = c_puct

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        # q-value
        if self.visit_count == 0:
            return 0
        # TODO müsste das nicht nur self.winner / self.visit_count sein?
        return self.value_sum / self.visit_count

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]

        # TODO: optimierung
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = uct_score(self, child, self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, go_game_state, to_play, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        self.to_play = to_play
        self.go_game_state = go_game_state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1, c_puct=self.c_puct)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.go_game_state.__str__(), prior, self.visit_count,
                                                         self.value())


class MCTS:

    def __init__(self, go_game_state, net_api, c_puct):
        self.go_game_state = go_game_state
        self.net_api = net_api
        # self.args = args
        self.c_puct = c_puct

    def run(self, net_api, go_game_state, to_play, new_root):

        if new_root is None:
            root = Node(0, to_play, self.c_puct)
        else:
            root = new_root
        # EXPAND root
        winner, action_probs = net_api.getPredictionFromNN(go_game_state.pos.board,
                                                           self.createHistoryStates(go_game_state),
                                                           go_game_state.pos.to_play)

        # print("self go game state")
        # print(self.go_game_state.pos.board)
        # print("self to play {}".format(self.go_game_state.pos.to_play))

        valid_moves = go_game_state.pos.all_legal_moves()
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        root.expand(go_game_state, go_game_state.pos.to_play, action_probs)

        for _ in range(constants.mcts_simulations):
            node = root
            search_path = [node]

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)

            parent = search_path[-2]
            go_game_state = parent
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            # next_go_game_state = self.go_game_state.move(action)  # return GoGamestate
            next_go_game_state = parent.go_game_state.move(action)  # return GoGamestate
            # Get the board from the perspective of the other player

            # TODO: müssen wir das machen oder reicht
            # next_state = self.go_game_state.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            winner = next_go_game_state.get_reward_for_player()

            if winner is None:
                # If the game has not ended:
                # EXPAND
                winner, action_probs = net_api.getPredictionFromNN(next_go_game_state.board,
                                                                   self.createHistoryStates(
                                                                       next_go_game_state),
                                                                   next_go_game_state.pos.to_play)
                # valid_moves = next_go_game_state.get_legal_actions()
                valid_moves = next_go_game_state.pos.all_legal_moves()
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs)
                node.expand(next_go_game_state, parent.go_game_state.pos.to_play * (-1), action_probs)
                # node.expand(next_go_game_state, next_go_game_state.pos.to_play, action_probs)

            # TODO: *(-1) korrekt?
            self.backpropagate(search_path, -winner, parent.go_game_state.pos.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def createHistoryStates(self, go_game_state):
        parentStates = []
        go_game_state.pos

        state_t_1 = None
        state_t_2 = None
        state_t_3 = None

        if len(go_game_state.pos.board_deltas) > 0:
            state_t_1 = go_game_state.pos.board - go_game_state.pos.board_deltas[0]
        if len(go_game_state.pos.board_deltas) > 1:
            state_t_2 = state_t_1 - go_game_state.pos.board_deltas[1]
            # state_t_2 = current_node.state.pos-current_node.state.pos.board_deltas[0]-current_node.state.pos.board_deltas[1] #equivalent
        if len(go_game_state.pos.board_deltas) > 2:
            state_t_3 = state_t_2 - go_game_state.pos.board_deltas[2]

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
