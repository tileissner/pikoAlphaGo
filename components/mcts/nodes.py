from random import random

import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod

from components.go.coords import from_flat


class MonteCarloTreeSearchNode(ABC):

    def __init__(self, state, parent=None):
        """
        Parameters
        ----------
        state : mctspy.games.common.TwoPlayersAbstractGameState
        parent : MonteCarloTreeSearchNode
        """
        self.state = state
        self.parent = parent
        self.children = []

    @property
    @abstractmethod
    def untried_actions(self):
        """

        Returns
        -------
        list of mctspy.games.common.AbstractGameAction

        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    # def is_fully_expanded(self):
    #     return len(self.untried_actions) == 0
    #entweder terminal node oder schon alle childs / moves angehängt
    def is_fully_expanded(self):
        #TODO umändern auf prüfen, ob alles auf 0 ist (1 = gültiger zug, 0 = ungültig)
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        #c_param = 0 --> exploitation
        #c_param irgendwas = exploration
        #choices weights = upper confidents bounds

        # Berechnung von u Wert nach Formel (--> Befragen des NN an dieser Stelle / hereingeben)
        #TODO: Verwendung der Knotenwerte die durch NN bereitgestellt werden + generell Formel anpassen (https://web.stanford.edu/~surag/posts/alphazero.html)
        choices_weights = []
        for c in self.children:
            if c.n == 0:
                c._number_of_visits = 1
            choices_weights.append((c.q_value / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n)))

        # choices_weights = [
        #     (c.q_value / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
        #     for c in self.children
        # ]
        # Zurückgeben der besten Aktion (enthalten in children) auf Basis d. berechneten u-wertes
        print("test")
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        indices = np.where(possible_moves == 1)
        return np.random.choice(indices[0])



class TwoPlayersGameMonteCarloTreeSearchNode(MonteCarloTreeSearchNode):

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None
        self.q_value = 0.
        self.p_distr = {}
        self.winner = None
        self.w_value = 0.



    @property
    # All Possible Moves --> Fill untried with lega actions.
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions


    @property
    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return self.w_value / self._number_of_visits

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        #print(self.untried_actions)
        print("untried actions")
        print(self.untried_actions)
        action = len(self.untried_actions) - 1 #Stack for tracking the possible actions at state
        self._untried_actions = np.delete(self.untried_actions, action)

        next_state = self.state.move(action)
        child_node = TwoPlayersGameMonteCarloTreeSearchNode(
            next_state, parent=self
        )
        child_node._number_of_visits = 0
        child_node.q_value = 0.0

        #p und v werte werden gesetzt
        self.children.append(child_node)
        return child_node

    def get_q_value(self):
        return self.w_value / (self._number_of_visits + 1.0) # TODO: +1.0 noch unklar?

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, w_value):

        self._number_of_visits += 1.
        self.w_value = self.w_value + w_value


        #self._results[result] += 1.

        # TODO: Set correct Q-Value


        if self.parent:
            self.parent.backpropagate(self.w_value)

