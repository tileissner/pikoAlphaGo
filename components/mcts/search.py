class MonteCarloTreeSearch(object):

    def __init__(self, node, pos, actions):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node


        #TODO Stimtm das so von der initialisierung?
        self.Q[pos.board, actions] = None
        self.P[pos.board, actions] = None
        self.v = None
        self.visitedNodes = []

    def best_action(self, simulations_number):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        Returns
        -------

        """
        # Actual MCTS Simulations (E
        for _ in range(0, simulations_number):
            v = self._tree_policy()
            reward = v.rollout()
            #TODO schätzung für reward muss von NN kommen --> Wird hochpropagiert
            v.backpropagate(reward)
            #search_function(self.)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def search_function(self, state, game, nnet):
        v = self._tree_policy()
        # Instead of Rollout: Pass v + q up along the search path
        # Ask neural net for board state returned by self.treepolicy
        # Except: Is terminal state, then propagate actual result
        reward = v.rollout()
        # TODO schätzung für das beste kind muss von NN kommen
        v.backpropagate(reward)

    def _tree_policy(self):
        """
        node selection
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        # TODO
        # fuer alphazero anpassen (nciht bis zur leaf node runtergehen)
        # Go DOES NOT expand until terminal node
        while not current_node.is_terminal_node():
            if self.node not in self.visitedNodes:
                self.visitedNodes.append(current_node)
                return current_node
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

